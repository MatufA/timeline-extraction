"""Behavioral coverage for native OmniTemp conversion using synthetic documents."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from timeline_extraction.data.omnitemp import prepare_omnitemp


@pytest.fixture
def native_document():
    return {
        "tokens": [
            "We", "took", "off", ",", "then", "took", "a", "break", ".",
            "Rain", "fell", ".",
        ],
        "allMentions": [
            {"tokens": "took", "eventIndex": 1, "m_id": "12", "tokens_ids": [5]},
            {"tokens": "took off", "eventIndex": 37, "m_id": "0", "tokens_ids": [1, 2]},
            {"tokens": "break", "eventIndex": 2, "m_id": 2, "tokens_ids": [7]},
            {"tokens": "fell", "eventIndex": 3, "m_id": "3", "tokens_ids": [10]},
        ],
        "allPairs": [
            {"_firstId": "12", "_secondId": 0, "_relation": "after"},
            {"_firstId": 2, "_secondId": "12", "_relation": "before"},
            {"_firstId": "0", "_secondId": "2", "_relation": "equal"},
        ],
    }


@pytest.fixture
def split_dir(tmp_path, native_document):
    path = tmp_path / "split"
    path.mkdir()
    (path / "synthetic.json").write_text(json.dumps(native_document), encoding="utf-8")
    return path


def test_span_markers_preserve_positions_multiword_events_and_punctuation(split_dir):
    gold, prepared = prepare_omnitemp(split_dir)

    assert list(gold.columns) == ["docid", "verb1", "verb2", "eiid1", "eiid2", "relation"]
    assert gold.values.tolist() == [
        ["synthetic", "took off", "break", "ei0", "ei2", "EQUAL"],
        ["synthetic", "took off", "took", "ei0", "ei12", "BEFORE"],
        ["synthetic", "break", "took", "ei2", "ei12", "BEFORE"],
    ]
    assert prepared["pair"][0] == {
        "text": "We [EVENT0]took off[/EVENT0] , then took a [EVENT2]break[/EVENT2] . Rain fell .",
        "relations": [["ei0", "ei2"]],
        "doc_id": "synthetic",
        "true_label": "equal",
    }
    assert prepared["pair"][2]["text"] == (
        "We took off , then [EVENT12]took[/EVENT12] a [EVENT2]break[/EVENT2] . Rain fell ."
    )
    assert prepared["multi"] == [{
        "text": (
            "We [EVENT0]took off[/EVENT0] , then [EVENT12]took[/EVENT12] a "
            "[EVENT2]break[/EVENT2] . Rain fell ."
        ),
        "relations": [["ei0", "ei2"], ["ei0", "ei12"], ["ei2", "ei12"]],
        "doc_id": "synthetic",
        "true_label": ["EQUAL", "BEFORE", "BEFORE"],
    }]
    # Unannotated pairs and the event with no annotated edges are not synthesized.
    assert len(prepared["pair"]) == 3
    assert "EVENT3" not in prepared["multi"][0]["text"]


@pytest.mark.parametrize("label,expected", [
    ("before", "AFTER"), ("after", "BEFORE"),
    ("equal", "EQUAL"), ("vague", "VAGUE"),
])
@pytest.mark.parametrize("first,second", [(12, "0"), ("12", 0)])
def test_reversed_numeric_pairs_preserve_label_meaning(
    tmp_path, native_document, label, expected, first, second,
):
    native_document["allPairs"] = [
        {"_firstId": first, "_secondId": second, "_relation": label}
    ]
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    gold, prepared = prepare_omnitemp(tmp_path)
    assert gold[["eiid1", "eiid2", "relation"]].values.tolist() == [["ei0", "ei12", expected]]
    assert prepared["pair"][0]["true_label"] == expected.lower()
    assert prepared["multi"][0]["true_label"] == [expected]


def test_duplicate_annotations_are_normalized_and_deduplicated(split_dir, native_document):
    baseline_gold, baseline_prepared = prepare_omnitemp(split_dir)
    native_document["allPairs"].extend([
        {"_firstId": 0, "_secondId": "12", "_relation": "before"},
        {"_firstId": "2", "_secondId": 0, "_relation": "equal"},
    ])
    (split_dir / "synthetic.json").write_text(json.dumps(native_document))
    gold, prepared = prepare_omnitemp(split_dir)
    pd.testing.assert_frame_equal(gold, baseline_gold)
    assert prepared == baseline_prepared


def test_conflicting_normalized_annotations_fail(split_dir, native_document):
    native_document["allPairs"].append(
        {"_firstId": 0, "_secondId": "12", "_relation": "after"}
    )
    (split_dir / "synthetic.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match="conflicting annotations.*ei0, ei12"):
        prepare_omnitemp(split_dir)


def test_output_order_is_stable_without_modifying_or_writing_input(tmp_path, native_document):
    # Creation order differs from filename order, and annotations differ in list order.
    (tmp_path / "z.json").write_text(json.dumps(native_document))
    reversed_document = deepcopy(native_document)
    reversed_document["allMentions"].reverse()
    reversed_document["allPairs"].reverse()
    (tmp_path / "a.json").write_text(json.dumps(reversed_document))
    (tmp_path / "README.txt").write_text("not a native document")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    gold, prepared = prepare_omnitemp(tmp_path)
    repeated_gold, repeated_prepared = prepare_omnitemp(tmp_path)

    assert gold.docid.tolist() == ["a"] * 3 + ["z"] * 3
    assert [record["doc_id"] for record in prepared["multi"]] == ["a", "z"]
    assert prepared["multi"][0]["text"] == prepared["multi"][1]["text"]
    assert prepared["multi"][0]["relations"] == prepared["multi"][1]["relations"]
    pd.testing.assert_frame_equal(repeated_gold, gold)
    assert repeated_prepared == prepared
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("event_id", [None, -1, True, 1.5, "ei0", "-1", "1.0", ""])
def test_invalid_event_ids_fail(tmp_path, native_document, event_id):
    native_document["allMentions"][1]["m_id"] = event_id
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match="event ID"):
        prepare_omnitemp(tmp_path)


@pytest.mark.parametrize("span,message", [
    ([], "nonempty list"), (None, "nonempty list"), ("1", "nonempty list"),
    ([1, 3], "ordered and contiguous"), ([2, 1], "ordered and contiguous"),
    ([1, 1], "ordered and contiguous"), ([-1], "in-range integers"),
    ([12], "in-range integers"), ([True], "in-range integers"),
    ([1.0], "in-range integers"), (["1"], "in-range integers"),
    ([5], "overlapping event spans"),
])
def test_malformed_or_unsupported_spans_fail(tmp_path, native_document, span, message):
    native_document["allMentions"][1]["tokens_ids"] = span
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match=message):
        prepare_omnitemp(tmp_path)


@pytest.mark.parametrize("pair,message", [
    ({"_firstId": 99, "_secondId": 0, "_relation": "before"}, "unknown event ID"),
    ({"_firstId": "e12", "_secondId": 0, "_relation": "before"}, "event ID"),
    ({"_firstId": True, "_secondId": 0, "_relation": "before"}, "event ID"),
    ({"_firstId": 0, "_secondId": "0", "_relation": "equal"}, "self-relations"),
    ({"_firstId": 12, "_secondId": 0, "_relation": "includes"}, "unsupported relation label"),
    ({"_firstId": 12, "_secondId": 0, "_relation": None}, "unsupported relation label"),
    ({"_firstId": 12, "_relation": "before"}, "event ID"),
    (None, "relation must be an object"),
])
def test_malformed_relations_fail_without_dropping_other_edges(tmp_path, native_document, pair, message):
    native_document["allPairs"].append(pair)
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match=message):
        prepare_omnitemp(tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("tokens", [], "tokens must be"), ("tokens", ["word", 1], "tokens must be"),
    ("tokens", [""], "tokens must be"), ("allMentions", None, "allMentions must be a list"),
    ("allMentions", [None], "mention must be an object"),
    ("allPairs", None, "allPairs must be a list"),
])
def test_malformed_document_fields_fail(tmp_path, native_document, field, value, message):
    native_document[field] = value
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match=message):
        prepare_omnitemp(tmp_path)


def test_duplicate_numeric_event_ids_fail(tmp_path, native_document):
    native_document["allMentions"].append(
        {"m_id": "00", "tokens_ids": [0], "tokens": "We"}
    )
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match="duplicate event ID 0"):
        prepare_omnitemp(tmp_path)


@pytest.mark.parametrize("content,message", [
    ("{not-json", "cannot read OmniTemp JSON"), ("[]", "document must be an object"),
    ("null", "document must be an object"),
])
def test_invalid_json_document_fails_even_with_valid_documents(split_dir, content, message):
    (split_dir / "broken.json").write_text(content)
    with pytest.raises(ValueError, match=message):
        prepare_omnitemp(split_dir)


def test_input_directory_must_exist_and_contain_json(tmp_path):
    with pytest.raises(ValueError, match="input directory does not exist"):
        prepare_omnitemp(tmp_path / "missing")
    with pytest.raises(ValueError, match="No OmniTemp JSON documents"):
        prepare_omnitemp(tmp_path)


def test_split_without_annotated_pairs_fails(tmp_path, native_document):
    native_document["allPairs"] = []
    (tmp_path / "doc.json").write_text(json.dumps(native_document))
    with pytest.raises(ValueError, match="No usable OmniTemp annotated pairs"):
        prepare_omnitemp(tmp_path)


def test_documents_without_pairs_do_not_invent_prepared_records(split_dir, native_document):
    native_document["allPairs"] = []
    (split_dir / "unannotated.json").write_text(json.dumps(native_document))
    gold, prepared = prepare_omnitemp(split_dir)
    assert gold.docid.tolist() == ["synthetic"] * 3
    assert [record["doc_id"] for record in prepared["multi"]] == ["synthetic"]
