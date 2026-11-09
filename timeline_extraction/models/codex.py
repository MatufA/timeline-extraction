"""Account-authenticated inference through the official Codex CLI JSONL protocol."""

import json

from timeline_extraction.models._cli import CLIModel


class Codex(CLIModel):
    """Use the CLI's cached ChatGPT login, never an API-key provider."""

    provider = "codex"
    executable = "codex"

    def _generate(self, prompt, workspace):
        args = [
            "exec", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
            "--json", "--model", self.model_name,
        ]
        # Keep CODEX_HOME/HOME for native credential storage, but ignore saved
        # provider, MCP, hook and agent settings. The built-in provider selects
        # the ChatGPT backend when the forced account login is active.
        settings = [
            'model_provider="openai"', 'forced_login_method="chatgpt"',
            'web_search="disabled"', "project_doc_max_bytes=0",
            "mcp_servers={}",
            "skills.bundled.enabled=false", "skills.include_instructions=false",
            "tools.update_plan.enabled=false",
            "tools.experimental_request_user_input.enabled=false",
        ]
        for feature in (
            "shell_tool", "shell_snapshot", "hooks", "apps", "plugins",
            "memories", "multi_agent", "multi_agent_v2", "view_image",
            "js_repl", "apply_patch_freeform",
        ):
            settings.append(f"features.{feature}=false")
        if self.reasoning_effort not in {None, "default"}:
            settings.append(f"model_reasoning_effort={json.dumps(self.reasoning_effort)}")
        for setting in settings:
            args.extend(["-c", setting])
        args.append("-")
        stdout = self._run(
            args, prompt, workspace,
            drop_env=(
                "CODEX_ACCESS_TOKEN", "OPENAI_FEDERATION_RULE_ID",
                "OPENAI_IDENTITY_TOKEN_FILE", "OPENAI_WORKLOAD_IDENTITY_CONTEXT",
                "CODEX_REFRESH_TOKEN_URL_OVERRIDE", "CODEX_REVOKE_TOKEN_URL_OVERRIDE",
                "CODEX_APP_SERVER_LOGIN_CLIENT_ID", "CODEX_AUTHAPI_BASE_URL",
            ),
        )
        return self._parse_output(stdout)

    @staticmethod
    def _parse_output(stdout):
        """Accept one completed turn, excluding reasoning and any tool activity.

        Codex JSONL has no message phase field. Its final-message writer uses
        the last completed agent_message in the successful turn, not reasoning
        summaries or earlier assistant commentary. Follow that same boundary.
        """
        started = False
        completed = False
        thread_seen = False
        response = None
        usage = None
        unfinished = set()
        finished = set()
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("codex CLI returned malformed JSONL") from error
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise ValueError("codex CLI returned an invalid JSONL event")
            kind = event["type"]
            if kind in {"error", "turn.failed", "turn.interrupted"}:
                detail = event.get("error", event.get("message", kind))
                raise RuntimeError(f"codex CLI generation failed: {str(detail)[-2000:]}")
            if completed:
                raise ValueError("codex CLI returned events after turn completion")
            if kind == "thread.started":
                if thread_seen or started:
                    raise ValueError("codex CLI returned multiple or misplaced threads")
                thread_seen = True
            elif kind == "turn.started":
                if started:
                    raise ValueError("codex CLI returned multiple turns")
                started = True
            elif kind == "turn.completed":
                if not started or unfinished:
                    raise ValueError("codex CLI completed an incomplete turn")
                completed = True
                usage = event.get("usage")
            elif kind in {"item.started", "item.updated", "item.completed"}:
                if not started:
                    raise ValueError("codex CLI returned an item outside a turn")
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ValueError("codex CLI returned an invalid item")
                item_type = item.get("type")
                if item_type == "error":
                    raise RuntimeError(
                        f"codex CLI generation failed: {str(item.get('message', 'item error'))[-2000:]}"
                    )
                # Not every built-in tool has a disable switch. Reject all tool
                # items (including future unknown kinds) rather than score a
                # classification contaminated by tool output.
                if item_type not in ("agent_message", "reasoning"):
                    raise ValueError(f"codex CLI used a tool or unsupported item: {item_type}")
                item_id = item.get("id")
                if not isinstance(item_id, str) or not item_id or item_id in finished:
                    raise ValueError("codex CLI returned an invalid item lifecycle")
                if kind == "item.started":
                    if item_id in unfinished:
                        raise ValueError("codex CLI returned a duplicate item start")
                    unfinished.add(item_id)
                elif kind == "item.updated":
                    if item_id not in unfinished:
                        raise ValueError("codex CLI updated an item without starting it")
                else:
                    if item.get("status") not in (None, "completed"):
                        raise ValueError("codex CLI returned an incomplete item")
                    unfinished.discard(item_id)
                    finished.add(item_id)
                    if item_type == "agent_message":
                        response = item.get("text")
            else:
                raise ValueError(f"codex CLI returned an unsupported event: {kind}")
        if not completed:
            raise ValueError("codex CLI did not complete its turn")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("codex CLI returned no final assistant text")
        if usage is not None and not isinstance(usage, dict):
            raise ValueError("codex CLI returned invalid usage")
        counts = {}
        for source, target in (
            ("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"),
        ):
            value = usage.get(source) if usage is not None else None
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"codex CLI returned invalid {source}")
            counts[target] = value
        prompt_tokens = counts["prompt_tokens"]
        completion_tokens = counts["completion_tokens"]
        # Cached input and reasoning output are subsets, not additional tokens.
        counts["total_tokens"] = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None else None
        )
        return {"response": response, **counts}
