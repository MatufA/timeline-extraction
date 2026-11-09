"""Tests for parser classes."""

import pytest

from timeline_extraction.models.LLModel import (
    LabelParser,
    JsonParser,
    NoParser,
    parse_dot_graph,
)


class TestParseDotGraph:
    """Test the parse_dot_graph function."""

    def test_parse_dot_graph_event_format(self):
        """Test parsing DOT graph with EVENT format."""
        dot_graph = 'EVENT1 -> EVENT2 [label="BEFORE"]'
        result = parse_dot_graph(dot_graph)

        assert len(result) == 1
        assert result[0]["event1"] == "EVENT1"
        assert result[0]["event2"] == "EVENT2"
        assert result[0]["relation"] == "BEFORE"

    def test_parse_dot_graph_e_format(self):
        """Test parsing DOT graph with e format."""
        dot_graph = 'e1 -> e2 [label="AFTER"]'
        result = parse_dot_graph(dot_graph)

        assert len(result) == 1
        assert result[0]["event1"] == "EVENT1"
        assert result[0]["event2"] == "EVENT2"
        assert result[0]["relation"] == "AFTER"

    def test_parse_dot_graph_empty(self):
        """Test parsing empty DOT graph."""
        result = parse_dot_graph("")
        assert result == []


class TestLabelParser:
    """Test the LabelParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = LabelParser()

    def test_parse_start_label(self):
        """Test parsing label at start of response."""
        response = {"response": "BEFORE the event occurred"}
        result = self.parser(response)

        assert result["response"] == "before"

    def test_parse_end_label(self):
        """Test parsing label at end of response."""
        response = {"response": "The relation is AFTER"}
        result = self.parser(response)

        assert result["response"] == "after"

    def test_parse_no_label(self):
        """Test parsing response with no valid label."""
        response = {"response": "This is not a temporal relation"}
        result = self.parser(response)

        assert result["response"] == "This is not a temporal relation"


class TestJsonParser:
    """Test the JsonParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = JsonParser()

    def test_parse_valid_json_list(self):
        """Test parsing valid JSON list response."""
        response = {
            "response": [{"event1": "EVENT1", "event2": "EVENT2", "relation": "BEFORE"}]
        }
        result = self.parser(response)

        assert len(result) == 1
        assert result[0]["relation"] == "BEFORE"

    def test_parse_json_string(self):
        """Test parsing JSON string response."""
        response = {
            "content": '[{"event1": "EVENT1", "event2": "EVENT2", "relation": "AFTER"}]'
        }
        result = self.parser(response)

        assert len(result) == 1
        assert result[0]["relation"] == "AFTER"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON response."""
        response = {"content": "invalid json content"}
        result = self.parser(response)

        assert result == []

    @pytest.mark.parametrize("response", [{"response": []}, {"content": "[]"}, {"response": None}])
    def test_empty_response(self, response):
        assert self.parser(response) == []

    @pytest.mark.parametrize("relation, expected", [
        ("BEFORE", "AFTER"), ("AFTER", "BEFORE"), ("EQUAL", "EQUAL"), ("VAGUE", "VAGUE"),
    ])
    def test_reversed_pair_preserves_temporal_meaning(self, relation, expected):
        result = self.parser({"response": [
            {"event1": "EVENT12", "event2": "EVENT2", "relation": relation}
        ]})
        assert result == [{"event1": "ei2", "event2": "ei12", "relation": expected}]

    def test_dot_graph_and_json_have_identical_semantics(self):
        dot = {"content": 'digraph { e12 -> e2 [label="before"]; }'}
        structured = {"response": [{"event1": 12, "event2": "ei2", "relation": "before"}]}
        assert self.parser(dot) == self.parser(structured)

    def test_fenced_json_preserves_justification(self):
        result = self.parser({"content": '```json\n[{"event1":1,"event2":2,"relation":"BEFORE","justification":"earlier"}]\n```'})
        assert result[0]["relation"] == "BEFORE"
        assert result[0]["justification"] == "earlier"

    def test_invalid_records_do_not_discard_valid_relations(self):
        result = self.parser({"response": [
            {}, None,
            {"event1": "bad", "event2": "ei2", "relation": "BEFORE"},
            {"event1": "ei1", "event2": "ei1", "relation": "BEFORE"},
            {"event1": "ei1", "event2": 2, "relation": "BEFORE"},
            {"event1": 1, "event2": "ei2", "relation": "before"},
        ]})
        assert result == [{"event1": "ei1", "event2": "ei2", "relation": "BEFORE"}]


class TestNoParser:
    """Test the NoParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = NoParser()

    def test_no_parsing(self):
        """Test that NoParser returns response unchanged."""
        response = {"response": "any content", "metadata": "test"}
        result = self.parser(response)

        assert result == response
