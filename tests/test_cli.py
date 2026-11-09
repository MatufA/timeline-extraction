"""Behavioral CLI regressions for research selection and configuration precedence."""

import csv
import json

import pytest
import yaml
from click.testing import CliRunner

from timeline_extraction.cli import cli
from timeline_extraction.config import ConfigManager, ModelConfig
from timeline_extraction.data.preprocessing import load_data


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


@pytest.fixture
def omnitemp_input(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    (source / "sample.json").write_text(json.dumps({
        "tokens": ["Dawn", "rose", "and", "rain", "fell", "then", "winds", "eased", "and", "stopped", "."],
        "allMentions": [
            {"tokens": "Dawn", "eventIndex": 0, "m_id": "99", "tokens_ids": [0]},
            {"tokens": "rose", "eventIndex": 1, "m_id": "10", "tokens_ids": [1]},
            {"tokens": "fell", "eventIndex": 2, "m_id": "2", "tokens_ids": [4]},
            {"tokens": "eased", "eventIndex": 3, "m_id": "0", "tokens_ids": [7]},
            {"tokens": "stopped", "eventIndex": 4, "m_id": "11", "tokens_ids": [9]},
        ],
        "allPairs": [
            {"_firstId": "10", "_secondId": 2, "_relation": "before"},
            {"_firstId": 2, "_secondId": "0", "_relation": "after"},
            {"_firstId": "10", "_secondId": 0, "_relation": "equal"},
            {"_firstId": 10, "_secondId": "11", "_relation": "vague"},
        ],
    }))
    return source


def test_prepare_omnitemp_exports_gold_and_both_research_modes(tmp_path, omnitemp_input):
    output = tmp_path / "prepared"
    config = tmp_path / "research.yaml"
    original = (omnitemp_input / "sample.json").read_bytes()
    runner = CliRunner()
    result = runner.invoke(cli, [
        "prepare-data", "--dataset-format", "omnitemp", "--input-dir", str(omnitemp_input),
        "--output-dir", str(output), "--output-config", str(config),
    ])
    assert result.exit_code == 0, result.output
    assert (omnitemp_input / "sample.json").read_bytes() == original
    gold_path = output / "omnitemp_gold.tsv"
    with gold_path.open() as file:
        rows = list(csv.reader(file, delimiter="\t"))
    expected = {
        ("ei2", "ei10"): ("fell", "rose", "AFTER"),
        ("ei0", "ei2"): ("eased", "fell", "BEFORE"),
        ("ei0", "ei10"): ("eased", "rose", "EQUAL"),
        ("ei10", "ei11"): ("rose", "stopped", "VAGUE"),
    }
    assert len(rows) == len(expected)
    assert all(len(row) == 6 and row[0] == "sample" for row in rows)
    assert {(row[3], row[4]): (row[1], row[2], row[5]) for row in rows} == expected
    assert len(load_data(gold_path)) == len(expected)

    pair_path = output / "pair_omnitemp_text_w_relations_prepared.json"
    multi_path = output / "multi_omnitemp_text_w_relations_prepared.json"
    pair = json.loads(pair_path.read_text())
    multi = json.loads(multi_path.read_text())
    assert len(pair) == len(expected)
    for record in pair:
        assert record["doc_id"] == "sample"
        assert len(record["relations"]) == 1
        relation = tuple(record["relations"][0])
        assert record["true_label"] == expected[relation][2].lower()
        assert record["text"].count("[EVENT") == 2
        assert record["text"].startswith("Dawn ")
        assert record["text"].endswith(".")
        for event, surface in zip(relation, expected[relation][:2]):
            assert f"[EVENT{event[2:]}]{surface}[/EVENT{event[2:]}]" in record["text"]
    assert len(multi) == 1
    assert multi[0]["doc_id"] == "sample"
    assert dict(zip(map(tuple, multi[0]["relations"]), multi[0]["true_label"])) == {
        relation: values[2] for relation, values in expected.items()
    }
    for event, surface in {"0": "eased", "2": "fell", "10": "rose", "11": "stopped"}.items():
        assert multi[0]["text"].count(f"[EVENT{event}]{surface}[/EVENT{event}]") == 1
    assert "[EVENT99]" not in multi[0]["text"]

    manager = ConfigManager(config)
    assert manager.research_config.dataset == "omnitemp"
    assert manager.data_config.datasets["omnitemp"] == {
        "gold_file": str(gold_path.resolve()),
        "prepared_files": {"pair": str(pair_path.resolve()), "multi": str(multi_path.resolve())},
    }
    plan_result = runner.invoke(cli, [
        "research", "--config-file", str(config), "--goal", "comparison", "--plan",
    ])
    assert plan_result.exit_code == 0, plan_result.output
    assert {row["experiment"]["mode"] for row in json.loads(plan_result.output)} == {"pair", "multi"}


@pytest.mark.parametrize("options", [
    [],
    ["--gold-file"],
    ["--docs-dir"],
    ["--input-dir"],
    ["--dataset-format", "omnitemp"],
    ["--dataset-format", "omnitemp", "--gold-file", "--input-dir"],
    ["--dataset-format", "omnitemp", "--docs-dir", "--input-dir"],
    ["--gold-file", "--docs-dir", "--input-dir"],
])
def test_prepare_data_rejects_missing_or_mixed_sources(tmp_path, omnitemp_input, options):
    gold = tmp_path / "gold.tsv"
    gold.write_text("sample\trose\tfell\t10\t2\tBEFORE\n")
    values = {"--gold-file": gold, "--docs-dir": omnitemp_input, "--input-dir": omnitemp_input}
    arguments = []
    for option in options:
        arguments.append(option)
        if option in values:
            arguments.append(str(values[option]))
    output = tmp_path / "prepared"
    config = tmp_path / "research.yaml"
    result = CliRunner().invoke(cli, [
        "prepare-data", *arguments, "--output-dir", str(output), "--output-config", str(config),
    ])
    assert result.exit_code != 0
    assert not output.exists()
    assert not config.exists()


@pytest.mark.parametrize("existing", [
    "omnitemp_gold.tsv",
    "pair_omnitemp_text_w_relations_prepared.json",
    "multi_omnitemp_text_w_relations_prepared.json",
])
def test_prepare_omnitemp_preflights_every_output(tmp_path, omnitemp_input, existing):
    output = tmp_path / "prepared"
    output.mkdir()
    existing_path = output / existing
    existing_path.write_bytes(b"preserve this artifact")
    config = tmp_path / "research.yaml"
    config.write_text("preserve this config")
    arguments = [
        "prepare-data", "--dataset-format", "omnitemp", "--input-dir", str(omnitemp_input),
        "--output-dir", str(output), "--output-config", str(config),
    ]
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code != 0
    assert existing_path.read_bytes() == b"preserve this artifact"
    assert set(output.iterdir()) == {existing_path}
    assert config.read_text() == "preserve this config"
    replaced = CliRunner().invoke(cli, [*arguments, "--overwrite"])
    assert replaced.exit_code == 0, replaced.output
    assert existing_path.read_bytes() != b"preserve this artifact"
    assert {path.name for path in output.iterdir()} == {
        "omnitemp_gold.tsv",
        "pair_omnitemp_text_w_relations_prepared.json",
        "multi_omnitemp_text_w_relations_prepared.json",
    }
