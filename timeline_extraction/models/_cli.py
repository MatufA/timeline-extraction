"""Shared process boundary for account-authenticated model CLIs."""

import abc
import json
import math
import os
import subprocess
from tempfile import TemporaryDirectory

from timeline_extraction.models.LLModel import LLModel, LabelParser
from timeline_extraction.models._reasoning import reasoning_options
from timeline_extraction.models._responses import chat_messages


class CLIModel(LLModel):
    """One isolated CLI session per trial; authentication remains owned by the CLI.

    CLI-managed generation does not expose API max_tokens or temperature controls.
    Non-default temperature is rejected rather than silently ignored.
    """

    provider: str
    executable: str

    def __init__(
        self, model_name, api_key=None, max_tokens=8192, temperature=0.0,
        n_trails=5, parser=LabelParser, *, reasoning_effort=None, cli_timeout=600,
    ):
        if api_key:
            raise ValueError(f"{self.provider} uses CLI account login, not api_key")
        if temperature != 0.0:
            raise ValueError(f"{self.provider} CLI does not support temperature")
        if not math.isfinite(cli_timeout) or cli_timeout <= 0:
            raise ValueError("cli_timeout must be a finite positive number of seconds")
        reasoning_options(self.provider, model_name, reasoning_effort)
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.cli_timeout = cli_timeout
        super().__init__(None, each_doc=True, n_trails=n_trails, parser=parser)

    def _run(self, args, prompt, workspace, *, drop_env=()):
        # Keep the native credential store, but never let API credentials switch
        # an account-backed run to metered API access. Do not expose tracing keys.
        excluded = {
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
            "LANGSMITH_API_KEY", "LANGFUSE_SECRET_KEY", *drop_env,
        }
        environment = {key: value for key, value in os.environ.items() if key not in excluded}
        try:
            result = subprocess.run(
                [self.executable, *args], input=prompt, text=True,
                encoding="utf-8", capture_output=True, cwd=workspace,
                env=environment, timeout=self.cli_timeout, check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"{self.executable} is not installed or not on PATH; install the official "
                f"{self.provider} CLI and sign in interactively before running experiments"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"{self.provider} CLI timed out after {self.cli_timeout:g} seconds"
            ) from error
        if result.returncode:
            detail = result.stderr.strip()[-2000:]
            # Native clients can report the real failure on stdout while stderr
            # contains only startup warnings. Extract errors, never partial text.
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                payload = event.get("result")
                payload = payload if isinstance(payload, dict) else event
                error = payload.get("error") or payload.get("errors")
                if not error and payload.get("type") == "error":
                    error = payload.get("message")
                if error:
                    detail += "\n" + str(error)[-2000:]
            raise RuntimeError(
                f"{self.provider} CLI exited with code {result.returncode}: {detail}"
            )
        return result.stdout

    def generate_response(self, prompt):
        messages = chat_messages(prompt)
        # Preserve system/user/few-shot boundaries without treating corpus text
        # as CLI flags, slash commands, shell syntax, or workspace instructions.
        text = (
            "Answer the final user message in the following JSON conversation. "
            "Respect its system instructions and examples. Return only the requested "
            "answer, without commentary. Do not use tools or inspect files.\n"
            + json.dumps(messages, ensure_ascii=False)
        )
        responses = []
        for _ in range(self.n_trails):
            with TemporaryDirectory(prefix=f"timeline-{self.provider}-") as workspace:
                responses.append(self._generate(text, workspace))
        return responses

    @abc.abstractmethod
    def _generate(self, prompt, workspace):
        """Return response text and optional canonical token counts for one trial."""
        raise NotImplementedError

    def prepare_response(self, responses):
        normalized = []
        for trail, response in enumerate(responses):
            text = response.get("response")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{self.provider} CLI returned no text content")
            normalized.append({
                "trail": trail, "response": text, "content": text,
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
                "total_tokens": response.get("total_tokens"),
            })
        return normalized
