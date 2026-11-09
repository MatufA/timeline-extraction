"""Research evaluation configuration, provider routing, and credential-free skip."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from timeline_extraction.config import ConfigManager, DataConfig, ModelConfig
from timeline_extraction.models.LLModel import LabelParser
from timeline_extraction.pipeline import (
    ModelFactory,
    PipelineExperimentConfig, TimelineExtractionPipeline,
)
from timeline_extraction.tracking import TrackingConfig


@pytest.fixture
def configured_pipeline(tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text('[]')
    gold = tmp_path / "gold.tsv"
    gold.write_text('doc\trun\tstop\t1\t2\tBEFORE\n')
    manager = ConfigManager()
    manager.data_config = DataConfig(
        data_path=str(tmp_path / "data"), results_path=str(tmp_path / "output"),
        datasets={"custom": {"prepared_file": str(prepared), "gold_file": str(gold)}},
    )
    manager.model_configs["openai"] = ModelConfig(name="model", provider="openai", n_trails=3)
    pipeline = TimelineExtractionPipeline(manager)
    config = PipelineExperimentConfig("model", "zero-shot", "pair", "custom", research_goal="classification")
    return pipeline, config, manager.model_configs["openai"]




def test_skip_evaluates_saved_responses_without_model_instantiation(monkeypatch, configured_pipeline):
    pipeline, config, model_config = configured_pipeline
    config = replace(config, skip_model_eval=True)
    paths = pipeline._get_file_paths(config, model_config)
    paths["llm_response"].parent.mkdir(parents=True)
    rows = [
        {"doc_id": "doc", "relations": [["ei1", "ei2"]], "response": "BEFORE",
         "content": "before", "trail": trial, "prompt": "stored prompt"}
        for trial in range(3)
    ]
    paths["llm_response"].write_text(''.join(json.dumps(row) + '\n' for row in rows))
    def forbidden(*args, **kwargs):
        raise AssertionError("skip must not instantiate a model or use credentials")
    monkeypatch.setattr(ModelFactory, "create_model", forbidden)
    result = pipeline.run_experiment(config, model_config)
    assert result.success, result.error
    voted = pd.read_csv(result.results_path)
    assert voted.p_label.tolist() == ["BEFORE"]
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["experiment_config"]["research_goal"] == "classification"
    assert metadata["model_config"]["n_trails"] == 3
    assert "api_key" not in metadata["model_config"]
    assert result.results_path.is_relative_to(pipeline.config_manager.data_config.results_path)


def test_skip_without_saved_predictions_fails_without_constructing_model(monkeypatch, configured_pipeline):
    pipeline, config, model_config = configured_pipeline
    monkeypatch.setattr(ModelFactory, "create_model", lambda *args: pytest.fail("model instantiated on skip"))
    result = pipeline.run_experiment(replace(config, skip_model_eval=True), model_config)
    assert not result.success
    assert "No saved responses" in result.error


def test_artifacts_isolate_provider_model_context_dataset_and_settings(configured_pipeline):
    pipeline, config, model = configured_pipeline
    original = pipeline._get_file_paths(config, model)["results"]
    variants = [
        (replace(config, full_context=False), model),
        (replace(config, mode="multi", parser_type="json"), model),
        (replace(config, model_name="organization/model"), replace(model, name="organization/model")),
        (config, replace(model, provider="groq")),
        (config, replace(model, temperature=0.5)),
        (config, replace(model, max_tokens=42)),
        (config, replace(model, n_trails=7)),
    ]
    assert all(pipeline._get_file_paths(changed, settings)["results"] != original for changed, settings in variants)
    assert pipeline._get_file_paths(replace(config, skip_model_eval=True, overwrite=True), model)["results"] == original
    assert pipeline._get_file_paths(config, replace(model, api_key="rotated"))["results"] == original
    pipeline._get_raw_text_path(config).write_text('[{"new":"input"}]')
    assert pipeline._get_file_paths(config, model)["results"] != original




@pytest.mark.parametrize("mode, content, expected", [
    ("pair", "BEFORE", "BEFORE"),
    ("multi", 'digraph { EVENT2 -> EVENT1 [label="AFTER"]; }', "BEFORE"),
])
def test_generated_pipeline_normalizes_and_resumes_without_remote_requests(
    monkeypatch, configured_pipeline, mode, content, expected,
):
    from tests.test_model_evaluation import LocalModel
    from timeline_extraction.models.LLModel import JsonParser

    pipeline, config, model_config = configured_pipeline
    config = replace(config, mode=mode, parser_type="label" if mode == "pair" else "json")
    pipeline._get_raw_text_path(config).write_text(json.dumps([{
        "doc_id": "doc", "text": "[EVENT1]run[/EVENT1] then [EVENT2]stop[/EVENT2]",
        "relations": [["ei1", "ei2"]], "true_label": "BEFORE",
    }]))
    parser = LabelParser if mode == "pair" else JsonParser
    model = LocalModel(parser=parser, content=content)
    monkeypatch.setattr(ModelFactory, "create_model", lambda *args: model)
    first = pipeline.run_experiment(config, model_config)
    assert first.success, first.error
    assert pd.read_csv(first.results_path).p_label.tolist() == [expected]
    assert model.calls == 1
    resumed = pipeline.run_experiment(config, model_config)
    assert resumed.success, resumed.error
    assert model.calls == 1
    assert first.results_path == resumed.results_path


def test_dataset_name_isolates_identical_source_files(configured_pipeline):
    pipeline, config, model_config = configured_pipeline
    pipeline.config_manager.data_config.datasets["custom2"] = dict(
        pipeline.config_manager.data_config.datasets["custom"]
    )
    assert pipeline._get_file_paths(config, model_config)["results"] != pipeline._get_file_paths(
        replace(config, data_name="custom2"), model_config
    )["results"]


@pytest.fixture
def langsmith_receiver():
    """Capture real SDK exports without contacting a hosted tracing service."""
    pytest.importorskip("langsmith")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    try:
        yield TrackingConfig(
            backend="langsmith", langsmith_api_key="local-tracking-secret",
            langsmith_endpoint=f"http://127.0.0.1:{server.server_port}",
        ), batches
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("mode, tracked, disconnected", [
    ("multi", False, False),
    ("pair", False, False),
    ("comb", False, False),
    ("multi", True, False),
    ("multi", False, True),
])
def test_cached_chain_scores_voted_explicit_and_expanded_artifacts(
    monkeypatch, configured_pipeline, request, mode, tracked, disconnected,
):
    pipeline, original_config, model_config = configured_pipeline
    pipeline.tracking_config = TrackingConfig(backend="none")
    if tracked:
        pipeline.tracking_config, batches = request.getfixturevalue("langsmith_receiver")
    config = replace(
        original_config, mode=mode, parser_type="json" if mode == "multi" else "label",
        skip_model_eval=True,
    )
    gold = (
        "doc\trun\tstop\t1\t2\tBEFORE\n"
        "doc\tstop\tleave\t2\t3\tBEFORE\n"
        "doc\trun\tleave\t1\t3\tBEFORE\n"
    )
    if disconnected:
        gold += "doc\trun\twait\t1\t4\tVAGUE\n"
    pipeline._get_labeled_data_path(config).write_text(gold)
    paths = pipeline._get_file_paths(config, model_config)
    assert paths["results"] == pipeline._get_file_paths(
        replace(config, skip_model_eval=False), model_config
    )["results"]
    paths["llm_response"].parent.mkdir(parents=True)
    # Each chain edge gets two votes, but the entire chain occurs in only one trial.
    # Expanding per trial before voting would therefore miss the transitive edge.
    trials = [
        [["ei1", "ei2"]],
        [["ei2", "ei3"]],
        [["ei1", "ei2"], ["ei2", "ei3"]],
    ]
    rows = [
        {"doc_id": "doc", "relations": relations, "response": "BEFORE",
         "true_label": "VAGUE", "content": "stored graph", "trail": trial,
         "prompt": "stored compact prompt"}
        for trial, relations in enumerate(trials)
    ]
    saved_responses = "".join(json.dumps(row) + "\n" for row in rows)
    paths["llm_response"].write_text(saved_responses)
    paths["checkpoint"].write_text('{"saved": "generation checkpoint"}\n')
    checkpoint = paths["checkpoint"].read_bytes()
    monkeypatch.setattr(
        ModelFactory, "create_model",
        lambda *args, **kwargs: pytest.fail("cached scoring must not construct a model"),
    )

    result = pipeline.run_experiment(config, model_config)
    assert result.success, result.error
    assert paths["llm_response"].read_text() == saved_responses
    assert paths["checkpoint"].read_bytes() == checkpoint
    parsed = pd.read_csv(paths["parsed_response"])
    assert len(parsed) == 4
    assert set(parsed.unique_id) == {"ei1-ei2", "ei2-ei3"}
    assert set(parsed.relation) == {"VAGUE"}  # Annotation never drives completion.
    voted = pd.read_csv(result.results_path)
    assert len(voted) == 2
    assert set(voted.unique_id) == {"ei1-ei2", "ei2-ei3"}
    assert voted.p_label.tolist() == ["BEFORE", "BEFORE"]
    assert "is_inferred" not in parsed and "is_inferred" not in voted
    metrics = pd.read_csv(result.metrics_path, header=[0, 1])
    assert metrics.columns.nlevels == 2
    names = set(metrics.columns.get_level_values(0))
    assert "accuracy" not in names and "BEFORE" not in names
    assert {"method", "suffix_path", "model_name"} <= names
    assert metrics["explicit.accuracy"].iloc[0, 0] == pytest.approx(2 / (3 + disconnected))
    assert metrics["explicit.coverage"].iloc[0, 0] == pytest.approx(2 / (3 + disconnected))
    assert metrics[("explicit.BEFORE", "precision")].iloc[0] == 1
    metadata = json.loads(result.metadata_path.read_text())
    if mode == "multi":
        assert result.expanded_results_path == paths["expanded_results"]
        expanded = pd.read_csv(result.expanded_results_path)
        assert expanded.columns.tolist() == [
            "docid", "eiid1", "eiid2", "p_label", "unique_id", "is_inferred",
        ]
        assert len(expanded) == 3
        assert expanded.set_index("unique_id").is_inferred.to_dict() == {
            "ei1-ei2": False, "ei2-ei3": False, "ei1-ei3": True,
        }
        assert set(expanded.p_label) == {"BEFORE"}
        assert metrics["expanded.accuracy"].iloc[0, 0] == pytest.approx(3 / (3 + disconnected))
        assert metrics["expanded.coverage"].iloc[0, 0] == pytest.approx(3 / (3 + disconnected))
        assert metadata["file_paths"]["expanded_results_path"] == str(result.expanded_results_path)
    else:
        assert result.expanded_results_path is None
        assert not paths["expanded_results"].exists()
        assert not any(name.startswith("expanded.") for name in names)
        assert metadata["file_paths"]["expanded_results_path"] is None


    if tracked:
        runs = {}
        for batch in batches:
            for run in batch.get("post", []) + batch.get("patch", []):
                runs.setdefault(run["id"], {}).update(run)
        root = next(run for run in runs.values() if run.get("name") == "timeline.experiment")
        assert not any(run.get("run_type") == "llm" for run in runs.values())
        output = root["outputs"]
        assert output["artifacts"]["expanded_results_path"] == str(result.expanded_results_path)
        assert output["artifacts"]["results_path"] == str(result.results_path)
        exported = output["metrics"][0]
        assert exported["explicit.accuracy"] == pytest.approx(2 / 3)
        assert exported["expanded.accuracy"] == 1
        assert exported["explicit.BEFORE.precision"] == metrics[("explicit.BEFORE", "precision")].iloc[0]
        assert exported["expanded.BEFORE.recall"] == metrics[("expanded.BEFORE", "recall")].iloc[0]
        assert "accuracy" not in exported


@pytest.mark.parametrize("mode", ["pair", "multi"])
@pytest.mark.parametrize("relations, response", [
    ([], ""),
    ([["ei1", "ei2"]], "NO_PREDS"),
])
def test_cached_empty_voted_predictions_keep_metrics_and_artifact_schema(
    monkeypatch, configured_pipeline, mode, relations, response,
):
    pipeline, original_config, model_config = configured_pipeline
    pipeline.tracking_config = TrackingConfig(backend="none")
    config = replace(
        original_config, mode=mode, parser_type="json" if mode == "multi" else "label",
        skip_model_eval=True,
    )
    paths = pipeline._get_file_paths(config, model_config)
    paths["llm_response"].parent.mkdir(parents=True)
    paths["llm_response"].write_text("".join(
        json.dumps({"doc_id": "doc", "relations": relations, "response": response,
                    "trail": trial, "prompt": "stored prompt"}) + "\n"
        for trial in range(3)
    ))
    monkeypatch.setattr(
        ModelFactory, "create_model",
        lambda *args, **kwargs: pytest.fail("cached scoring must not construct a model"),
    )
    result = pipeline.run_experiment(config, model_config)
    assert result.success, result.error
    assert pd.read_csv(result.results_path).empty
    metrics = pd.read_csv(result.metrics_path, header=[0, 1])
    for view in (["explicit", "expanded"] if mode == "multi" else ["explicit"]):
        assert metrics[f"{view}.accuracy"].iloc[0, 0] == 0
        assert metrics[f"{view}.coverage"].iloc[0, 0] == 0
    if mode == "multi":
        expanded = pd.read_csv(result.expanded_results_path)
        assert expanded.empty
        assert expanded.columns.tolist() == [
            "docid", "eiid1", "eiid2", "p_label", "unique_id", "is_inferred",
        ]
    else:
        assert result.expanded_results_path is None
        assert not paths["expanded_results"].exists()
