"""Antigravity account inference via the official headless ``agy`` CLI.

Protocol: https://antigravity.google/docs/cli/headless
Agent schema: https://antigravity.google/docs/subagents
"""

import json
from pathlib import Path

from timeline_extraction.models._cli import CLIModel


class Antigravity(CLIModel):
    provider = "antigravity"
    executable = "agy"

    def _generate(self, prompt, workspace):
        # A dedicated tool-free primary agent avoids inheriting the interactive
        # coding agent's shell, filesystem, browser, MCP, or subagent capabilities.
        agents = Path(workspace) / ".agents" / "agents"
        agents.mkdir(parents=True)
        (agents / "timeline-classifier.md").write_text(
            "---\nname: timeline-classifier\n"
            "description: Classify temporal relations from supplied text only.\n"
            "tools: []\nmcpServers: []\nskills: []\nplugins: []\n"
            "mainAgent: true\nsubagent: false\ncommandExecutionPolicy: 'off'\n"
            "---\nFollow the supplied conversation and return only its requested answer.\n",
            encoding="utf-8",
        )
        args = [
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--disable-slash-commands", "--agent", "timeline-classifier",
            "--model", self.model_name, "--sandbox",
            "--print-timeout", f"{self.cli_timeout:g}s",
            "--settings", json.dumps({
                "modelProvider": "",
                "allowNonWorkspaceAccess": False,
                "permissions": {"deny": [
                    "read_file(*)", "write_file(*)", "command(*)",
                    "read_url(*)", "execute_url(*)", "mcp(*)",
                ]},
            }),
        ]
        if self.reasoning_effort not in {None, "default"}:
            args += ["--effort", self.reasoning_effort]
        stdout = self._run(
            args, json.dumps({"event": "user", "message": {"content": prompt}}) + "\n",
            workspace, drop_env=("AGY_ADC_AUTH", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GEMINI_BASE_URL"),
        )
        result = None
        for line in stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Antigravity CLI returned an invalid event")
            if event.get("event") == "step_update":
                step = event.get("step_update", {})
                if step.get("step_type") == "tool":
                    raise ValueError("Antigravity attempted tool use during classification")
            if event.get("event") == "result":
                if result is not None:
                    raise ValueError("Antigravity returned multiple turns for one trial")
                result = event.get("result")
                if not isinstance(result, dict) or result.get("status") != "SUCCESS":
                    detail = result.get("error") if isinstance(result, dict) else "invalid result"
                    raise ValueError(f"Antigravity did not complete the response: {detail}")
        if result is None:
            raise ValueError("Antigravity CLI returned no completed result")
        usage = result.get("usage") or {}
        return {
            "response": result.get("response"),
            "prompt_tokens": usage.get("input_tokens"),
            # Antigravity output_tokens already includes thinking_tokens.
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
