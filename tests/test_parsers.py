"""Tests for parser classes."""

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
        assert result[0]["event1"] == 1
        assert result[0]["event2"] == 2
        assert result[0]["relation"] == "BEFORE"

    def test_parse_json_string(self):
        """Test parsing JSON string response."""
        response = {
            "content": '[{"event1": "EVENT1", "event2": "EVENT2", "relation": "AFTER"}]'
        }
        result = self.parser(response)

        assert len(result) == 1
        assert result[0]["event1"] == 1
        assert result[0]["event2"] == 2
        assert result[0]["relation"] == "AFTER"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON response."""
        response = {"content": "invalid json content"}
        result = self.parser(response)

        assert result == []


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
