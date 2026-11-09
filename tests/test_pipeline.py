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
