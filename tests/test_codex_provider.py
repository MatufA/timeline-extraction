"""Official Codex JSONL contract regressions; no account or network calls."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from timeline_extraction.models.codex import Codex


def item(kind="agent_message", text="before", item_id="final", **extra):
    return {"id": item_id, "type": kind, "text": text, **extra}


def events(*items, usage=None):
    return [
        {"type": "thread.started", "thread_id": "fresh-thread"},
        {"type": "turn.started"},
        *({"type": "item.completed", "item": value} for value in items),
        {"type": "turn.completed", "usage": usage},
    ]


def jsonl(values):
    return "\n".join(json.dumps(value) for value in values)


def test_final_assistant_not_reasoning_or_commentary_and_no_double_counting():
    result = Codex._parse_output(jsonl(events(
        item("reasoning", "Private reasoning", "reasoning"),
        item(text="Working on the answer", item_id="commentary"),
        item(text="before"),
        usage={
            "input_tokens": 100, "cached_input_tokens": 80,
            "cache_write_input_tokens": 10, "output_tokens": 25,
            "reasoning_output_tokens": 20,
        },
    )))
    assert result == {
        "response": "before", "prompt_tokens": 100,
        "completion_tokens": 25, "total_tokens": 125,
    }


@pytest.mark.parametrize("usage,expected", [
    (None, (None, None, None)),
    ({}, (None, None, None)),
    ({"input_tokens": 8}, (8, None, None)),
    ({"output_tokens": 3}, (None, 3, None)),
    ({"input_tokens": 0, "output_tokens": 0}, (0, 0, 0)),
])
def test_unknown_usage_stays_unknown_and_real_zero_stays_zero(usage, expected):
    result = Codex._parse_output(jsonl(events(item(), usage=usage)))
    assert tuple(result[key] for key in (
        "prompt_tokens", "completion_tokens", "total_tokens",
    )) == expected


def test_absent_usage_is_not_fabricated():
    stream = events(item())
    stream[-1].pop("usage")
    result = Codex._parse_output(jsonl(stream))
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["total_tokens"] is None


@pytest.mark.parametrize("usage", [
    [], "usage", {"input_tokens": -1}, {"output_tokens": -1},
    {"input_tokens": True}, {"output_tokens": 2.5}, {"input_tokens": "12"},
])
def test_invalid_usage_cannot_be_recorded(usage):
    with pytest.raises(ValueError, match="invalid"):
        Codex._parse_output(jsonl(events(item(), usage=usage)))


@pytest.mark.parametrize("event", [
    {"type": "error", "message": "account expired"},
    {"type": "turn.failed", "error": {"message": "rate limit"}},
    {"type": "turn.interrupted"},
    {"type": "item.completed", "item": item("error", message="model rerouted")},
])
def test_failure_even_after_assistant_text_is_not_a_classification(event):
    stream = events(item())
    stream.insert(-1, event)
    with pytest.raises(RuntimeError, match="generation failed"):
        Codex._parse_output(jsonl(stream))


@pytest.mark.parametrize("kind", [
    "command_execution", "file_change", "mcp_tool_call", "collab_tool_call",
    "web_search", "todo_list", "future_tool_kind",
])
@pytest.mark.parametrize("lifecycle", ["item.started", "item.updated", "item.completed"])
def test_any_tool_activity_invalidates_the_classification(kind, lifecycle):
    stream = events(item())
    stream.insert(2, {"type": lifecycle, "item": item(kind)})
    with pytest.raises(ValueError, match="tool or unsupported item"):
        Codex._parse_output(jsonl(stream))


@pytest.mark.parametrize("stream", [
    [],
    events(item())[:-1],
    events(item("reasoning", "before")),
    events(item(text="")),
    events(item(text=" \n")),
    events(item(text=None)),
    events(item(status="incomplete")),
    events(item(status="failed")),
    [{"type": "turn.completed"}],
    [{"type": "item.completed", "item": item()}, {"type": "turn.completed"}],
    events(item()) + [{"type": "turn.completed"}],
    events(item()) + [{"type": "turn.started"}],
    [{"type": "turn.started"}, *events(item())],
    [*events(item())[:2], {"type": "item.started", "item": item()},
     {"type": "turn.completed"}],
    [{"type": "turn.started"}, {"type": "item.completed", "item": None}],
    [{"type": "turn.started"}, {"type": "item.completed", "item": {"type": "agent_message"}}],
    [{"type": "turn.started"}, {"type": "item.updated", "item": item()}],
    [{"type": "unexpected.event"}],
    [None], ["before"], [{}], [{"type": []}],
])
def test_incomplete_malformed_or_ambiguous_streams_are_not_classifications(stream):
    with pytest.raises(ValueError):
        Codex._parse_output(jsonl(stream))


def test_completed_message_lifecycle_excludes_partial_text():
    stream = events()
    stream[2:2] = [
        {"type": "item.started", "item": item(text="par")},
        {"type": "item.updated", "item": item(text="partial")},
        {"type": "item.completed", "item": item(text="before")},
    ]
    assert Codex._parse_output(jsonl(stream))["response"] == "before"


def test_non_json_stdout_is_not_an_answer():
    with pytest.raises(ValueError, match="malformed JSONL"):
        Codex._parse_output(jsonl(events(item())) + "\nnot json")


def test_fresh_trials_enforce_account_backend_and_preserve_native_login(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    forbidden = (
        "OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_ACCESS_TOKEN",
        "OPENAI_FEDERATION_RULE_ID", "OPENAI_IDENTITY_TOKEN_FILE",
        "OPENAI_WORKLOAD_IDENTITY_CONTEXT", "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
        "CODEX_REVOKE_TOKEN_URL_OVERRIDE", "CODEX_APP_SERVER_LOGIN_CLIENT_ID",
        "CODEX_AUTHAPI_BASE_URL",
    )
    for key in forbidden:
        monkeypatch.setenv(key, "must-not-be-inherited")
    calls = []

    def run(command, **kwargs):
        workspace = Path(kwargs["cwd"])
        assert workspace.is_dir()
        assert not tuple(workspace.iterdir())
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0, stderr="", stdout=jsonl(events(
                item(text="before" if len(calls) == 1 else "after"),
                usage={"input_tokens": len(calls) * 10, "output_tokens": len(calls)},
            )),
        )

    monkeypatch.setattr("timeline_extraction.models._cli.subprocess.run", run)
    model = Codex("gpt-6.1-sol", reasoning_effort="high", n_trails=2, cli_timeout=42)
    prompt = [
        {"role": "system", "content": "Classify the relation."},
        {"role": "user", "content": "Text with --resume, $(shell), and unicode é"},
    ]
    results = model.prepare_response(model.generate_response(prompt))
    assert [(row["trail"], row["response"], row["total_tokens"]) for row in results] == [
        (0, "before", 11), (1, "after", 22),
    ]
    assert calls[0][1]["cwd"] != calls[1][1]["cwd"]
    for command, kwargs in calls:
        assert command[:2] == ["codex", "exec"]
        assert command[-1] == "-"
        assert command[command.index("--model") + 1] == "gpt-6.1-sol"
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert {"--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
                "--ephemeral", "--json"}.issubset(command)
        settings = dict(command[index + 1].split("=", 1)
                        for index, value in enumerate(command) if value == "-c")
        assert settings["forced_login_method"] == '"chatgpt"'
        assert settings["model_provider"] == '"openai"'
        assert settings["model_reasoning_effort"] == '"high"'
        assert settings["web_search"] == '"disabled"'
        assert settings["mcp_servers"] == "{}"
        assert settings["project_doc_max_bytes"] == "0"
        for feature in ("shell_tool", "hooks", "apps", "plugins", "multi_agent", "view_image"):
            assert settings[f"features.{feature}"] == "false"
        assert not any("bypass" in value or "resume" == value for value in command)
        assert "shell" not in kwargs
        assert kwargs["timeout"] == 42
        assert kwargs["env"]["HOME"] == str(tmp_path / "home")
        assert kwargs["env"]["CODEX_HOME"] == str(tmp_path / "codex-home")
        assert not set(forbidden).intersection(kwargs["env"])
        assert kwargs["input"].endswith(json.dumps(prompt, ensure_ascii=False))
        assert "$(shell)" not in command
        assert not Path(kwargs["cwd"]).exists()


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing"])
def test_process_failures_propagate_without_scoring_partial_stdout(monkeypatch, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "missing":
            raise FileNotFoundError(command[0])
        return SimpleNamespace(returncode=1, stderr="account login required", stdout=jsonl(events(item())))

    monkeypatch.setattr("timeline_extraction.models._cli.subprocess.run", run)
    model = Codex("gpt-6.1-sol", n_trails=1)
    with pytest.raises(RuntimeError):
        model.generate_response("Classify this sentence")


@pytest.mark.parametrize("effort", [None, "default"])
def test_default_reasoning_uses_cli_default_without_invalid_literal(monkeypatch, effort):
    captured = []

    def run(args, prompt, workspace, **kwargs):
        captured.extend(args)
        return jsonl(events(item()))

    model = Codex("gpt-6.1-sol", n_trails=1, reasoning_effort=effort)
    monkeypatch.setattr(model, "_run", run)
    assert model.generate_response("Classify this sentence")[0]["response"] == "before"
    assert not any(value.startswith("model_reasoning_effort=") for value in captured)


def test_empty_final_message_does_not_fall_back_to_earlier_commentary():
    stream = events(item(text="before", item_id="commentary"), item(text=""))
    with pytest.raises(ValueError, match="no final assistant text"):
        Codex._parse_output(jsonl(stream))


def test_error_after_completion_invalidates_the_whole_generation():
    stream = events(item()) + [{"type": "error", "message": "shutdown failed"}]
    with pytest.raises(RuntimeError, match="shutdown failed"):
        Codex._parse_output(jsonl(stream))


def test_terminal_error_is_not_masked_by_startup_warning(monkeypatch):
    stdout = jsonl([
        {"type": "turn.started"},
        {"type": "item.completed", "item": item(text="partial classification")},
        {"type": "turn.failed", "error": {"message": "Account cannot use this model"}},
    ])
    monkeypatch.setattr(
        "timeline_extraction.models._cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="models cache warning", stdout=stdout,
        ),
    )
    with pytest.raises(RuntimeError, match="Account cannot use this model") as error:
        Codex("gpt-6.1-sol", n_trails=1).generate_response("classify")
    assert "partial classification" not in str(error.value)
