"""Native Antigravity terminal states must not become invalid predictions."""

import json
import subprocess

import pytest

from timeline_extraction.models.antigravity import Antigravity


def run_events(monkeypatch, tmp_path, events):
    model = Antigravity("gemini-4-argon-high", n_trails=1, reasoning_effort="high")
    monkeypatch.setattr(model, "_run", lambda *args, **kwargs: "\n".join(json.dumps(event) for event in events))
    return model.prepare_response([model._generate("classify", str(tmp_path))])


def test_terminal_answer_excludes_stream_deltas_and_counts_thinking_once(monkeypatch, tmp_path):
    result = run_events(monkeypatch, tmp_path, [
        {"event": "init", "init": {"tools": []}},
        {"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "not the final answer"}},
        {"event": "result", "result": {
            "status": "SUCCESS", "response": 'digraph { EVENT1 -> EVENT2 [label="before"]; }',
            "usage": {"input_tokens": 20, "output_tokens": 12, "thinking_tokens": 8,
                      "cache_read_tokens": 10, "total_tokens": 32},
        }},
    ])[0]
    assert result["response"] == 'digraph { EVENT1 -> EVENT2 [label="before"]; }'
    assert (result["prompt_tokens"], result["completion_tokens"], result["total_tokens"]) == (20, 12, 32)


@pytest.mark.parametrize("status", ["ERROR", "CANCELED", "INTERRUPTED", "INVALID", "WAITING", "RUNNING", None])
def test_non_success_result_cannot_be_scored(monkeypatch, tmp_path, status):
    with pytest.raises(ValueError, match="did not complete"):
        run_events(monkeypatch, tmp_path, [{"event": "result", "result": {"status": status, "response": "BEFORE"}}])


@pytest.mark.parametrize("events", [
    [],
    [{"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "BEFORE"}}],
    [{"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "run_command"}}],
    [{"event": "result", "result": {"status": "SUCCESS", "response": "BEFORE"}}] * 2,
    [{"event": "result", "result": {"status": "SUCCESS", "response": ""}}],
    [{"event": "result", "result": {"status": "SUCCESS"}}],
    [{"event": "result", "result": []}],
])
def test_missing_ambiguous_or_tool_results_fail_closed(monkeypatch, tmp_path, events):
    with pytest.raises(ValueError):
        run_events(monkeypatch, tmp_path, events)


def test_absent_usage_remains_unknown(monkeypatch, tmp_path):
    result = run_events(monkeypatch, tmp_path, [
        {"event": "result", "result": {"status": "SUCCESS", "response": "BEFORE"}},
    ])[0]
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["total_tokens"] is None


def test_process_timeout_does_not_return_a_prediction(monkeypatch, tmp_path):
    model = Antigravity("gemini-4-argon-high", n_trails=1, cli_timeout=0.1)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("agy", 0.1, output="BEFORE")

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        model._generate("classify", str(tmp_path))
