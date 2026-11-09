"""Tests for data preprocessing functions."""

import json

import pytest

from timeline_extraction.data.preprocessing import (
    replace_eid, Doc, load_data, generate_training_dataset, prepare_as_jsonl,
)


class TestReplaceEid:
    """Test the replace_eid function."""

    def test_replace_eid_basic(self):
        """Test basic event ID replacement."""
        text = "ei1:event1 and ei2:event2 occurred"
        exclude_ids = []
        result = replace_eid(text, exclude_ids)

        assert "ei1:event1" not in result
        assert "ei2:event2" not in result
        assert "event1" in result
        assert "event2" in result

    def test_replace_eid_with_exclusion(self):
        """Test event ID replacement with exclusions."""
        text = "ei1:event1 and ei2:event2 occurred"
        exclude_ids = ["ei1"]
        result = replace_eid(text, exclude_ids)

        assert "[EVENT1]event1[/EVENT1]" in result
        assert "ei2:event2" not in result  # Should be replaced
        assert "event2" in result

    def test_replace_eid_no_matches(self):
        """Test replace_eid with no event IDs."""
        text = "This is just regular text"
        exclude_ids = []
        result = replace_eid(text, exclude_ids)

        assert result == text

    def test_replace_eid_empty_text(self):
        """Test replace_eid with empty text."""
        text = ""
        exclude_ids = []
        result = replace_eid(text, exclude_ids)

        assert result == ""


@pytest.fixture
def timeml_path(tmp_path):
    path = tmp_path / "doc.tml"
    path.write_text("""
    <TimeML>
      <DOCID>doc</DOCID>
      <TEXT>They <EVENT eid="e12">stopped</EVENT> after they <EVENT eid="e2">started</EVENT>.</TEXT>
      <MAKEINSTANCE eventID="e12" eiid="ei12"/>
      <MAKEINSTANCE eventID="e2" eiid="ei2"/>
      <TLINK eventInstanceID="ei12" relatedToEventInstance="ei2" relType="AFTER"/>
    </TimeML>
    """)
    return path


class TestDoc:
    def test_loads_timeml_event_instances_and_text(self, timeml_path):
        doc = Doc(timeml_path)
        assert doc.docid == "doc"
        assert doc.mapping == {"e12": "ei12", "e2": "ei2"}
        assert "ei12:stopped" in doc.text
        assert "ei2:started" in doc.text
        assert doc.get_text() == doc.text

    def test_reads_directed_temporal_links(self, timeml_path):
        assert Doc(timeml_path).get_relations() == [("ei12", "ei2", "AFTER")]


@pytest.mark.parametrize("mode", ["pair", "multi"])
def test_gold_to_prepared_text_preserves_numeric_pair_label(timeml_path, tmp_path, mode):
    gold_path = tmp_path / "gold.tsv"
    gold_path.write_text("doc\tstopped\tstarted\t12\t2\tAFTER\n")
    gold = load_data(gold_path)
    prepared = generate_training_dataset(gold, timeml_path.parent, mode)
    destination = tmp_path / f"{mode}.json"
    prepare_as_jsonl(prepared, destination, mode)
    records = json.loads(destination.read_text())
    assert len(records) == 1
    assert records[0]["relations"] == [["ei2", "ei12"]]
    assert records[0]["true_label"] == ("before" if mode == "pair" else ["BEFORE"])
    assert "[EVENT12]stopped[/EVENT12]" in records[0]["text"]
    assert "[EVENT2]started[/EVENT2]" in records[0]["text"]


def test_gold_after_label_keeps_original_pair_meaning(tmp_path):
    path = tmp_path / "gold.tsv"
    path.write_text("doc\tfirst\tsecond\t2\t12\tAFTER\n")
    gold = load_data(path)
    assert gold.loc[0, ["eiid1", "eiid2", "label", "relation"]].tolist() == [
        "ei2", "ei12", "AFTER", "AFTER",
    ]
