"""Opt-in tracking must not turn offline commands into credential/network clients."""

import builtins
import json

import pytest
from click.testing import CliRunner

from timeline_extraction.cli import cli
from timeline_extraction.tracking import TrackingConfig


def test_cli_backend_overrides_environment_before_validating_credentials(monkeypatch):
    monkeypatch.setenv("TIMELINE_TRACKING", "langsmith")
    for key in ("LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    command = ["model", "evaluate", "--provider", "openai", "--model-name", "model", "--mode", "pair", "--data-name", "te3-platinum"]
    env_selected = CliRunner().invoke(cli, command)
    assert env_selected.exit_code != 0
    assert "LangSmith tracking requires" in env_selected.output
    cli_selected = CliRunner().invoke(cli, ["--tracking", "langfuse", *command])
    assert cli_selected.exit_code != 0
    assert "Langfuse tracking requires" in cli_selected.output
    assert "LangSmith tracking requires" not in cli_selected.output


def forbid_tracking_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"langsmith", "langfuse"}:
            raise AssertionError("Offline commands must not load a tracking SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("backend", ["langsmith", "langfuse", "both"])
def test_enabled_tracking_plan_requires_neither_sdks_nor_credentials(monkeypatch, backend):
    forbid_tracking_imports(monkeypatch)
    result = CliRunner().invoke(cli, ["--tracking", backend, "research", "--plan"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["experiment"]["research_goal"] == "classification"


def test_explicit_none_runs_without_sdk_despite_enabled_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMELINE_TRACKING", "both")
    forbid_tracking_imports(monkeypatch)
    result = CliRunner().invoke(cli, [
        "--tracking", "none", "model", "evaluate", "--provider", "openai",
        "--model-name", "model", "--mode", "pair", "--data-name", "te3-platinum",
        "--skip-model-eval", "--output-dir", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert "No saved responses to evaluate" in result.output
    assert "tracking requires" not in result.output


def test_programmatic_environment_selection_can_be_explicitly_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TIMELINE_TRACKING", "both")
    forbid_tracking_imports(monkeypatch)
    from timeline_extraction.pipeline import PipelineExperimentConfig, TimelineExtractionPipeline

    pipeline = TimelineExtractionPipeline(tracking_config=TrackingConfig(backend="none"))
    result = pipeline.run_experiment(PipelineExperimentConfig(
        "model", "zero-shot", "pair", "te3-platinum", skip_model_eval=True,
        output_dir=str(tmp_path),
    ))
    assert not result.success
    assert "No saved responses to evaluate" in result.error


def test_langsmith_exports_trial_usage_and_generation_failures():
    pytest.importorskip("langsmith")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from tests.test_model_evaluation import LocalModel
    from timeline_extraction.tracking import ExperimentTracker

    batches = []

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond({"batch_ingest_config": {
                "use_multipart_endpoint": False, "size_limit": 100,
                "scale_up_nthreads_limit": 1, "scale_up_qsize_trigger": 1000,
                "scale_down_nempty_trigger": 4,
            }})

        def do_POST(self):
            batches.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.respond({})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    tracker = ExperimentTracker(TrackingConfig(
        backend="langsmith", langsmith_api_key="tracking-secret",
        langsmith_endpoint=f"http://127.0.0.1:{server.server_port}",
    ))
    metadata = {"ls_provider": "local", "ls_model_name": "local-model"}
    try:
        with tracker.span("experiment", {}, {}) as experiment:
            tracker.generate(LocalModel(), "Classify these events", metadata)
            with pytest.raises(RuntimeError, match="request interrupted"):
                tracker.generate(LocalModel(fail_after=0), "Classify these events", metadata)
            experiment.finish({"success": True})
        tracker.flush()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    runs = {}
    for batch in batches:
        for run in batch.get("post", []) + batch.get("patch", []):
            runs.setdefault(run["id"], {}).update(run)
    root = next(run for run in runs.values() if run.get("name") == "experiment")
    generations = [run for run in runs.values() if run.get("run_type") == "llm"]
    completed = next(run for run in generations if not run.get("error"))
    assert completed["outputs"]["usage_metadata"] == {
        "input_tokens": 21, "output_tokens": 9, "total_tokens": 30,
    }
    assert all(run["parent_run_id"] == root["id"] for run in generations)
    assert any("request interrupted" in run.get("error", "") for run in generations)
