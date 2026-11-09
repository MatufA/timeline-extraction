"""Provider-neutral persistence and resume behavior, without remote requests."""

import json

import pytest

from timeline_extraction.models.LLModel import LLModel, JsonParser, LabelParser


class Prompt:
    def generate_dict_prompt(self, text, relations):
        return [{"role": "user", "content": text}]


class LocalModel(LLModel):
    def __init__(self, parser=JsonParser, n_trails=3, content=None, fail_after=None):
        super().__init__(None, each_doc=False, n_trails=n_trails, parser=parser)
        self.content = content if content is not None else '[{"event1":1,"event2":2,"relation":"BEFORE"},{"event1":2,"event2":3,"relation":"AFTER"}]'
        self.calls = 0
        self.fail_after = fail_after

    def generate_response(self, prompt):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("request interrupted")
        return [self.content] * self.n_trails

    def prepare_response(self, responses):
        return [
            {"trail": trial, "response": content, "content": content,
             "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            for trial, content in enumerate(responses)
        ]


def evaluate(model, tmp_path, records, checkpoint=True, overwrite=False):
    text_path = tmp_path / "input.json"
    text_path.write_text(json.dumps(records))
    return model.generate_responses(
        text_path, tmp_path / "results.jsonl", Prompt(), ["text", "relations"],
        overwrite=overwrite,
        checkpoint_results=tmp_path / "checkpoint.jsonl" if checkpoint else None,
    )


def record(text="context", relations=None):
    return {"doc_id": "doc", "text": text, "relations": relations or [["ei1", "ei2"], ["ei2", "ei3"]]}


def test_independent_rows_and_provider_neutral_raw_checkpoint(tmp_path):
    model = LocalModel()
    results = evaluate(model, tmp_path, [record()])
    assert len(results) == 6
    assert [row["response"] for row in results[:2]] == ["BEFORE", "AFTER"]
    results[0]["relations"][0][0] = "changed"
    assert results[1]["relations"] == [["ei2", "ei3"]]
    checkpoints = [json.loads(line) for line in (tmp_path / "checkpoint.jsonl").read_text().splitlines()]
    assert len(checkpoints) == 3
    assert checkpoints[0]["content"] == model.content
    assert checkpoints[0]["response"] == model.content
    assert checkpoints[0]["total_tokens"] == 10
    assert checkpoints[0]["prompt"] == [{"role": "user", "content": "context"}]


@pytest.mark.parametrize("checkpoint", [True, False])
def test_resume_retains_every_relation_row_and_trial(tmp_path, checkpoint):
    original = evaluate(LocalModel(), tmp_path, [record()], checkpoint=checkpoint)
    resumed_model = LocalModel()
    resumed = evaluate(resumed_model, tmp_path, [record()], checkpoint=checkpoint)
    assert resumed_model.calls == 0
    assert resumed == original


def test_resume_identity_includes_context_and_all_pairs(tmp_path):
    records = [record("same text"), record("same text", [["ei1", "ei2"], ["ei2", "ei4"]]), record("other text")]
    model = LocalModel()
    results = evaluate(model, tmp_path, records)
    assert model.calls == 3
    assert len({row["request_key"] for row in results}) == 3
    resumed_model = LocalModel()
    assert evaluate(resumed_model, tmp_path, records) == results
    assert resumed_model.calls == 0


def test_checkpoint_recovers_completed_requests_after_interruption(tmp_path):
    records = [record(), record("second context")]
    with pytest.raises(RuntimeError, match="request interrupted"):
        evaluate(LocalModel(fail_after=1), tmp_path, records)
    resumed_model = LocalModel()
    resumed = evaluate(resumed_model, tmp_path, records)
    assert resumed_model.calls == 1
    assert len(resumed) == 12
    assert len({(row["request_key"], row["trail"]) for row in resumed}) == 6


def test_partial_trial_checkpoint_fills_only_missing_trials(tmp_path):
    original = evaluate(LocalModel(), tmp_path, [record()])
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(checkpoint.read_text().splitlines()[0] + "\n")
    (tmp_path / "results.jsonl").unlink()
    resumed_model = LocalModel()
    resumed = evaluate(resumed_model, tmp_path, [record()])
    assert resumed_model.calls == 1
    assert resumed == original
    assert len(checkpoint.read_text().splitlines()) == 3


def test_empty_parsed_trials_are_completed_in_checkpoint(tmp_path):
    assert evaluate(LocalModel(content="[]"), tmp_path, [record()]) == []
    resumed_model = LocalModel(content="[]")
    assert evaluate(resumed_model, tmp_path, [record()]) == []
    assert resumed_model.calls == 0


def test_overwrite_replaces_old_checkpoint_and_predictions(tmp_path):
    evaluate(LocalModel(parser=LabelParser, content="BEFORE"), tmp_path, [record()])
    changed = LocalModel(parser=LabelParser, content="AFTER")
    results = evaluate(changed, tmp_path, [record()], overwrite=True)
    assert changed.calls == 1
    assert all(row["response"] == "after" for row in results)
    assert len((tmp_path / "checkpoint.jsonl").read_text().splitlines()) == 3
