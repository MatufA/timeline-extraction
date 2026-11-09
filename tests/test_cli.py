"""Behavioral CLI regressions for research selection and configuration precedence."""

import json

import yaml
from click.testing import CliRunner

from timeline_extraction.cli import cli
from timeline_extraction.config import ConfigManager, ModelConfig


def test_comparison_plan_preserves_provider_and_method_controls(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "models": {
            "together": {"default_model": "openai/gpt-oss-120b", "api_key": "private-key", "n_trails": 3},
            "groq": {"default_model": "openai/gpt-oss-20b", "n_trails": 1},
        },
        "research": {"goal": "comparison", "providers": ["together", "groq"], "methods": ["zero-shot", "few-shot"], "dataset": "custom"},
    }))
    result = CliRunner().invoke(cli, ["--config", str(config), "research", "--plan"])
    assert result.exit_code == 0, result.output
    assert "private-key" not in result.output
    plan = json.loads(result.output)
    controls = {(row["model"]["provider"], row["experiment"]["method"], row["experiment"]["mode"]) for row in plan}
    assert controls == {(p, m, mode) for p in ("together", "groq") for m in ("zero-shot", "few-shot") for mode in ("pair", "multi")}
    for row in plan:
        experiment, model = row["experiment"], row["model"]
        assert experiment["data_name"] == "custom"
        assert experiment["parser_type"] == ("label" if experiment["mode"] == "pair" else "json")
        assert model["n_trails"] == (3 if model["provider"] == "together" else 1)


def test_command_configuration_merges_nested_provider_settings(tmp_path):
    global_file = tmp_path / "global.yaml"
    global_file.write_text(yaml.safe_dump({"models": {"openai": {"default_model": "custom-model", "max_tokens": 9000, "n_trails": 3}}}))
    local_file = tmp_path / "local.yaml"
    local_file.write_text(yaml.safe_dump({"models": {"openai": {"temperature": 0.5}}}))
    result = CliRunner().invoke(cli, ["--config", str(global_file), "research", "--config-file", str(local_file), "--plan"])
    assert result.exit_code == 0, result.output
    model = json.loads(result.output)[0]["model"]
    assert (model["name"], model["max_tokens"], model["n_trails"], model["temperature"]) == ("custom-model", 9000, 3, 0.5)


def test_custom_model_cannot_silently_route_to_multiple_providers():
    result = CliRunner().invoke(cli, ["research", "--provider", "openai", "--provider", "groq", "--model-name", "custom", "--plan"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_research_preflights_all_inputs_before_provider_creation(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"data": {"data_path": str(tmp_path / "missing"), "results_path": str(tmp_path / "results")}}))
    result = CliRunner().invoke(cli, ["research", "--config-file", str(config)])
    assert result.exit_code != 0
    assert "Missing input:" in result.output
    assert "API key" not in result.output


def test_configuration_round_trip_preserves_experiment_controls(tmp_path):
    manager = ConfigManager()
    manager.add_model_config("groq", ModelConfig(name="custom/model", provider="groq", max_tokens=7000, temperature=0.4, n_trails=3))
    manager.evaluation_config.method = "few-shot"
    manager.evaluation_config.mode = "multi"
    manager.research_config.goal = "comparison"
    manager.research_config.providers = ["groq"]
    manager.save_config(tmp_path / "experiment.yaml")
    loaded = ConfigManager(tmp_path / "experiment.yaml")
    assert loaded.resolve_model_config(provider="groq") == manager.resolve_model_config(provider="groq")
    assert loaded.evaluation_config == manager.evaluation_config
    assert loaded.research_config == manager.research_config


def test_prepare_data_exports_usable_pair_and_document_inputs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.tml").write_text(
        '<TimeML><DOCID>sample</DOCID><TEXT>'
        '<EVENT eid="e1">arrived</EVENT> then <EVENT eid="e2">left</EVENT>.'
        '</TEXT><MAKEINSTANCE eventID="e1" eiid="ei1"/>'
        '<MAKEINSTANCE eventID="e2" eiid="ei2"/></TimeML>'
    )
    gold = tmp_path / "gold.txt"
    gold.write_text("sample\tarrived\tleft\t1\t2\tBEFORE\n")
    output = tmp_path / "prepared"
    config = tmp_path / "research.yaml"
    result = CliRunner().invoke(cli, [
        "prepare-data", "--gold-file", str(gold), "--docs-dir", str(docs),
        "--data-name", "custom", "--output-dir", str(output), "--output-config", str(config),
    ])
    assert result.exit_code == 0, result.output
    loaded = ConfigManager(config)
    assert loaded.research_config.dataset == "custom"
    paths = loaded.data_config.datasets["custom"]
    assert paths["gold_file"] == str(gold)
    pair = json.loads((output / "pair_custom_text_w_relations_prepared.json").read_text())
    multi = json.loads((output / "multi_custom_text_w_relations_prepared.json").read_text())
    assert pair[0]["relations"] == multi[0]["relations"] == [["ei1", "ei2"]]
    assert pair[0]["true_label"] == "before"
    assert "[EVENT1]arrived[/EVENT1]" in pair[0]["text"]
    assert "[EVENT2]left[/EVENT2]" in multi[0]["text"]
    second = CliRunner().invoke(cli, [
        "prepare-data", "--gold-file", str(gold), "--docs-dir", str(docs),
        "--data-name", "custom", "--output-dir", str(output),
    ])
    assert second.exit_code != 0
    assert json.loads((output / "pair_custom_text_w_relations_prepared.json").read_text()) == pair
