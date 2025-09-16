"""Tests for data preprocessing functions."""

from timeline_extraction.data.preprocessing import replace_eid, Doc


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

        assert "ei1:event1" in result  # Should not be replaced
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


class TestDoc:
    """Test the Doc class."""

    def test_doc_initialization(self):
        """Test Doc class initialization."""
        doc = Doc(
            doc_id="test_doc",
            text="Test text with ei1:event1",
            events={"ei1": "event1"},
            relations=[],
        )

        assert doc.doc_id == "test_doc"
        assert doc.text == "Test text with ei1:event1"
        assert doc.events == {"ei1": "event1"}
        assert doc.relations == []

    def test_doc_get_relations(self):
        """Test getting relations from Doc."""
        relations = [{"event1": "ei1", "event2": "ei2", "relation": "BEFORE"}]
        doc = Doc(
            doc_id="test_doc",
            text="Test text",
            events={"ei1": "event1", "ei2": "event2"},
            relations=relations,
        )

        result = doc.get_relations()
        assert len(result) == 1
        assert result[0]["relation"] == "BEFORE"

    def test_doc_from_file_mock(self):
        """Test Doc.from_file method (mock test)."""
        # This would require a mock TML file or actual test data
        # For now, we'll test the method exists
        assert hasattr(Doc, "from_file")
        assert callable(Doc.from_file)
