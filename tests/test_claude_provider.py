"""Deterministic Claude Code account/protocol tests; no CLI or hosted calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from timeline_extraction.models.claude import Claude


def subscription_status(**overrides):
    return {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty",
            **overrides}


def successful_result(**overrides):
    return {
        "type": "result", "subtype": "success", "is_error": False,
        "result": "before", "stop_reason": "end_turn", "terminal_reason": "completed",
        "num_turns": 1, "permission_denials": [],
        "usage": {
            "input_tokens": 10, "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30, "output_tokens": 4,
            "cache_creation": {"ephemeral_5m_input_tokens": 20},
        },
        **overrides,
    }


def stub_cli(monkeypatch, model, result=None, status=None):
    calls = []

    def run(args, prompt, workspace, *, drop_env=()):
        calls.append((args, prompt, workspace, drop_env))
        if "auth" in args:
            return json.dumps(subscription_status() if status is None else status)
        return json.dumps(successful_result() if result is None else result)

    monkeypatch.setattr(model, "_run", run)
    return calls


def test_constructor_is_offline_and_missing_cli_is_actionable(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("claude")

    monkeypatch.setattr("timeline_extraction.models._cli.subprocess.run", missing)
    model = Claude("claude-sonnet-4-6", n_trails=1)
    with pytest.raises(RuntimeError, match="install.*Claude Code.*claude auth login"):
        model.generate_response("classify this")


def test_subscription_inference_is_isolated_and_preserves_roles(monkeypatch):
    model = Claude("claude-sonnet-4-6", n_trails=2, reasoning_effort="high")
    calls = stub_cli(monkeypatch, model)
    prompt = [
        {"role": "system", "content": "Return a relation."},
        {"role": "assistant", "content": "after"},
        {"role": "user", "content": "--resume /danger @private.txt"},
    ]
    rows = model.prepare_response(model.generate_response(prompt))
    assert len(calls) == 4
    for auth, inference in (calls[:2], calls[2:]):
        assert auth[0][-2:] == ["auth", "status"]
        assert auth[1] == ""
        args, stdin, workspace, _ = inference
        assert json.loads(stdin.split("\n", 1)[1]) == prompt
        assert "@private.txt" not in args
        assert workspace == auth[2]
        assert not Path(workspace).exists()
        for flag, value in (
            ("--input-format", "text"), ("--output-format", "json"),
            ("--model", "claude-sonnet-4-6"), ("--effort", "high"),
            ("--tools", ""), ("--setting-sources", ""),
            ("--permission-mode", "default"),
        ):
            assert args[args.index(flag) + 1] == value
        assert "--safe-mode" in args
        assert "--strict-mcp-config" in args
        assert json.loads(args[args.index("--mcp-config") + 1]) == {"mcpServers": {}}
        settings = json.loads(args[args.index("--settings") + 1])
        assert settings["forceLoginMethod"] == "claudeai"
        assert settings["disableAllHooks"] is True
        assert settings["disableClaudeAiConnectors"] is True
        assert settings["env"]["CLAUDE_CODE_DISABLE_ATTACHMENTS"] == "1"
        assert "--disable-slash-commands" in args
        assert "--no-session-persistence" in args
        assert "--no-chrome" in args
        assert not {"--bare", "--resume", "--continue", "--dangerously-skip-permissions",
                    "--allow-dangerously-skip-permissions"}.intersection(args)
    assert calls[1][2] != calls[3][2]
    assert rows == [
        {"trail": trail, "response": "before", "content": "before",
         "prompt_tokens": 60, "completion_tokens": 4, "total_tokens": 64}
        for trail in range(2)
    ]


@pytest.mark.parametrize("status", [
    subscription_status(loggedIn=False),
    subscription_status(authMethod="api_key"),
    subscription_status(authMethod="api_key_helper"),
    subscription_status(authMethod="oauth_token"),
    subscription_status(authMethod="third_party"),
    subscription_status(apiProvider="bedrock"),
    subscription_status(apiProvider="vertex"),
    {},
])
def test_non_subscription_login_never_generates(monkeypatch, tmp_path, status):
    model = Claude("claude-sonnet-4-6", n_trails=1)
    calls = stub_cli(monkeypatch, model, status=status)
    with pytest.raises(RuntimeError, match="cached claude.ai subscription login"):
        model._generate("classify", str(tmp_path))
    assert len(calls) == 1


@pytest.mark.parametrize("overrides,error", [
    ({"type": "assistant"}, "failed"),
    ({"subtype": "error_max_turns", "errors": ["turn budget exhausted"]}, "turn budget"),
    ({"subtype": "error_during_execution", "errors": ["Login expired"]}, "Login expired"),
    ({"subtype": "error_max_budget_usd"}, "failed"),
    ({"is_error": True, "result": "authentication failed"}, "authentication failed"),
    ({"stop_reason": "max_tokens"}, "stop_reason"),
    ({"stop_reason": "refusal"}, "stop_reason"),
    ({"stop_reason": "tool_use"}, "stop_reason"),
    ({"stop_reason": None}, "stop_reason"),
    ({"terminal_reason": "aborted_streaming"}, "incomplete terminal"),
    ({"terminal_reason": "hook_stopped"}, "incomplete terminal"),
    ({"terminal_reason": "model_error"}, "incomplete terminal"),
    ({"num_turns": 0}, "non-inference"),
    ({"permission_denials": [{"tool_name": "Read"}]}, "blocked"),
    ({"api_error_status": 401}, "blocked"),
    ({"result": " "}, "no text"),
    ({"result": None}, "no text"),
    ({"usage": []}, "invalid usage"),
])
def test_failed_or_incomplete_results_are_not_classifications(monkeypatch, tmp_path, overrides, error):
    model = Claude("claude-sonnet-4-6", n_trails=1)
    stub_cli(monkeypatch, model, result=successful_result(**overrides))
    with pytest.raises(ValueError, match=error):
        model._generate("classify", str(tmp_path))


def test_older_protocol_without_stop_reason_fails_closed(monkeypatch, tmp_path):
    model = Claude("claude-sonnet-4-6", n_trails=1)
    result = successful_result()
    result.pop("stop_reason")
    stub_cli(monkeypatch, model, result=result)
    with pytest.raises(ValueError, match="update Claude Code"):
        model._generate("classify", str(tmp_path))


@pytest.mark.parametrize("usage,expected", [
    (None, (None, None, None)),
    ({}, (None, None, None)),
    ({"input_tokens": 2, "output_tokens": 3}, (None, 3, None)),
    ({"input_tokens": 2, "cache_creation_input_tokens": None,
      "cache_read_input_tokens": 0, "output_tokens": 3}, (None, 3, None)),
    ({"input_tokens": 2, "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0}, (2, None, None)),
    ({"input_tokens": 0, "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0, "output_tokens": 0}, (0, 0, 0)),
])
def test_unknown_usage_is_not_zero(monkeypatch, tmp_path, usage, expected):
    model = Claude("claude-sonnet-4-6", n_trails=1)
    stub_cli(monkeypatch, model, result=successful_result(usage=usage))
    result = model._generate("classify", str(tmp_path))
    assert tuple(result[key] for key in (
        "prompt_tokens", "completion_tokens", "total_tokens",
    )) == expected


@pytest.mark.parametrize("name", [
    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens",
])
@pytest.mark.parametrize("value", [-1, True, 1.5, "10"])
def test_malformed_token_counts_fail(monkeypatch, tmp_path, name, value):
    model = Claude("claude-sonnet-4-6", n_trails=1)
    usage = successful_result()["usage"]
    usage[name] = value
    stub_cli(monkeypatch, model, result=successful_result(usage=usage))
    with pytest.raises(ValueError, match=f"usage.{name}"):
        model._generate("classify", str(tmp_path))


@pytest.mark.parametrize("bad_output", ["not json", "[]", "null"])
@pytest.mark.parametrize("auth_output", [False, True])
def test_malformed_protocol_is_not_text(monkeypatch, tmp_path, bad_output, auth_output):
    model = Claude("claude-sonnet-4-6", n_trails=1)

    def run(args, *unused, **kwargs):
        if auth_output or "auth" not in args:
            return bad_output
        return json.dumps(subscription_status())

    monkeypatch.setattr(model, "_run", run)
    with pytest.raises(ValueError, match="invalid"):
        model._generate("classify", str(tmp_path))


def test_process_drops_injected_credentials_and_provider_overrides(monkeypatch, tmp_path):
    removed = (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS", "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_EFFORT_LEVEL",
        "CLAUDE_CODE_FORCE_SESSION_PERSISTENCE",
    )
    for key in removed:
        monkeypatch.setenv(key, "injected")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "native-login"))
    processes = []

    def subprocess_run(argv, **kwargs):
        processes.append((argv, kwargs))
        assert not set(removed).intersection(kwargs["env"])
        assert kwargs["env"]["HOME"] == str(tmp_path)
        assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "native-login")
        assert kwargs.get("shell", False) is False
        assert kwargs["cwd"] == str(tmp_path)
        payload = subscription_status() if "auth" in argv else successful_result()
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("timeline_extraction.models._cli.subprocess.run", subprocess_run)
    model = Claude("claude-sonnet-4-6", n_trails=1)
    model._generate("classify", str(tmp_path))
    assert len(processes) == 2
    assert all(argv[0] == "claude" for argv, _ in processes)
    assert processes[0][1]["input"] == ""
    assert processes[1][1]["input"] == "classify"


def test_native_login_error_mentions_subscription_setup(monkeypatch, tmp_path):
    model = Claude("claude-sonnet-4-6", n_trails=1)

    def unauthenticated(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout='{"loggedIn":false}', stderr="Not logged in")

    monkeypatch.setattr("timeline_extraction.models._cli.subprocess.run", unauthenticated)
    with pytest.raises(RuntimeError, match="claude auth login.*not --console"):
        model._generate("classify", str(tmp_path))
