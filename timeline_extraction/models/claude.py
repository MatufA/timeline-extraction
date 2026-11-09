"""Tool-free Claude Code print sessions using its cached claude.ai login."""

import json

from timeline_extraction.models._cli import CLIModel


# Keep HOME and CLAUDE_CONFIG_DIR: Claude, not this adapter, owns credentials.
# Remove alternate providers, injected credentials, and overrides that outrank flags.
_DROP_ENV = (
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_MANTLE", "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN", "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_SESSION_ACCESS_TOKEN", "CLAUDE_CODE_API_KEY",
    "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_PROFILE",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_AWS_BASE_URL", "ANTHROPIC_AWS_WORKSPACE_ID",
    "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
    "ANTHROPIC_FOUNDRY_BASE_URL", "ANTHROPIC_FOUNDRY_RESOURCE",
    "ANTHROPIC_BEDROCK_BASE_URL", "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL", "CLAUDE_CODE_SIMPLE",
    "CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_FORCE_SESSION_PERSISTENCE",
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
)

_SETTINGS = json.dumps({
    "forceLoginMethod": "claudeai",
    "disableAllHooks": True,
    "disableClaudeAiConnectors": True,
    "autoMemoryEnabled": False,
    "env": {
        "CLAUDE_CODE_AUTO_CONNECT_IDE": "false",
        "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    },
})


def _json_object(stdout, description):
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"claude CLI returned invalid {description} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"claude CLI returned invalid {description} object")
    return value


def _token_count(usage, name):
    value = usage.get(name)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"claude CLI returned invalid usage.{name}")
    return value


class Claude(CLIModel):
    """Invoke the official CLI without reading or exporting account credentials.

    Requires a current Claude Code with auth status and result stop_reason support.
    Managed organization policy still applies; it cannot be disabled by this adapter.
    """

    provider = "claude"
    executable = "claude"

    def _generate(self, prompt: str, workspace: str) -> dict:
        # --bare is deliberately NOT used: it disables cached OAuth/keychain login.
        # Safe mode preserves auth but excludes skills, plugins, memory and prompts.
        isolation = ["--safe-mode", "--setting-sources", "", "--settings", _SETTINGS]
        try:
            status_text = self._run(
                [*isolation, "auth", "status"], "", workspace, drop_env=_DROP_ENV,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "claude subscription login check failed; install the official Claude Code "
                "CLI and run `claude auth login` with a claude.ai subscription account "
                f"(not --console): {error}"
            ) from error
        status = _json_object(status_text, "authentication status")
        if (status.get("loggedIn") is not True
                or status.get("authMethod") != "claude.ai"
                or status.get("apiProvider") != "firstParty"):
            raise RuntimeError(
                "claude requires a cached claude.ai subscription login, not an API key, "
                "Console profile, injected OAuth token, or cloud provider; "
                "run `claude auth login` (not --console)"
            )

        args = [
            *isolation, "--print", "--input-format", "text", "--output-format", "json",
            "--model", self.model_name, "--tools", "", "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands",
            "--no-session-persistence", "--no-chrome", "--permission-mode", "default",
        ]
        if self.reasoning_effort not in {None, "default"}:
            args.extend(["--effort", self.reasoning_effort])
        result = _json_object(
            self._run(args, prompt, workspace, drop_env=_DROP_ENV), "result",
        )
        if (result.get("type") != "result" or result.get("subtype") != "success"
                or result.get("is_error") is not False):
            detail = result.get("errors") or result.get("result") or result.get("subtype")
            raise ValueError(f"claude CLI failed: {str(detail)[-2000:]}")
        # A success subtype alone can also represent refusals, truncation or hooks.
        # Fail closed on older protocols that cannot establish a complete answer.
        if result.get("stop_reason") != "end_turn":
            raise ValueError(
                "claude CLI did not complete normally: "
                f"stop_reason={result.get('stop_reason')!r}; "
                "update Claude Code if this field is missing"
            )
        if result.get("terminal_reason", "completed") != "completed":
            raise ValueError(f"claude CLI incomplete terminal: {result['terminal_reason']}")
        if (type(result.get("num_turns")) is not int or result["num_turns"] < 1
                or result.get("permission_denials") or result.get("api_error_status")):
            raise ValueError("claude CLI returned a blocked or non-inference result")
        text = result.get("result")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("claude CLI returned no text content")

        usage = result.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ValueError("claude CLI returned invalid usage object")
        usage = usage or {}
        inputs = [_token_count(usage, key) for key in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        )]
        # Anthropic input_tokens excludes cache writes and reads. Missing/null
        # components are unknown, not zero; nested cache breakdowns are not added.
        prompt_tokens = sum(inputs) if all(value is not None for value in inputs) else None
        completion_tokens = _token_count(usage, "output_tokens")
        total_tokens = (prompt_tokens + completion_tokens
                        if prompt_tokens is not None and completion_tokens is not None else None)
        return {
            "response": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
