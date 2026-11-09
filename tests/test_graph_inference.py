"""Gold-independent temporal completion of voted document graphs."""

from copy import deepcopy

import pandas as pd
import pytest

from timeline_extraction.graph import Graph


COLUMNS = ["docid", "eiid1", "eiid2", "p_label", "unique_id", "is_inferred"]


def voted(rows):
    return pd.DataFrame(rows, columns=["docid", "eiid1", "eiid2", "p_label"])


def relations(frame):
    assert frame.columns.tolist() == COLUMNS
    assert frame["is_inferred"].dtype == bool
    assert not frame.duplicated(["docid", "eiid1", "eiid2"]).any()
    return {
        (docid, first, second): (label, inferred)
        for docid, first, second, label, _, inferred in frame.itertuples(index=False, name=None)
    }


def expand(rows):
    return Graph(relation_key="p_label").generate_implicit_relations(voted(rows))


def test_before_chain_completes_every_reachable_pair_with_provenance():
    result = expand([
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "BEFORE"),
        ("doc", "ei3", "ei4", "BEFORE"),
    ])
    assert relations(result) == {
        ("doc", "ei1", "ei2"): ("BEFORE", False),
        ("doc", "ei1", "ei3"): ("BEFORE", True),
        ("doc", "ei1", "ei4"): ("BEFORE", True),
        ("doc", "ei2", "ei3"): ("BEFORE", False),
        ("doc", "ei2", "ei4"): ("BEFORE", True),
        ("doc", "ei3", "ei4"): ("BEFORE", False),
    }


def test_reversed_after_uses_numeric_pair_order_but_lexical_unique_ids():
    result = expand([
        ("doc", "ei10", "ei2", "AFTER"),
        ("doc", "ei2", "ei0", "AFTER"),
    ])
    assert result.values.tolist() == [
        ["doc", "ei0", "ei2", "BEFORE", "ei0-ei2", False],
        ["doc", "ei0", "ei10", "BEFORE", "ei0-ei10", True],
        ["doc", "ei2", "ei10", "BEFORE", "ei10-ei2", False],
    ]


def test_after_reachability_retains_the_canonical_direction():
    result = expand([
        ("doc", "ei0", "ei2", "AFTER"),
        ("doc", "ei2", "ei10", "AFTER"),
    ])
    assert relations(result) == {
        ("doc", "ei0", "ei2"): ("AFTER", False),
        ("doc", "ei0", "ei10"): ("AFTER", True),
        ("doc", "ei2", "ei10"): ("AFTER", False),
    }


def test_equality_only_chains_complete_each_group_without_conflating_groups():
    result = expand([
        ("doc", "ei0", "ei2", "EQUAL"),
        ("doc", "ei2", "ei10", "EQUAL"),
        ("doc", "ei3", "ei4", "EQUAL"),
        ("doc", "ei4", "ei5", "EQUAL"),
    ])
    assert relations(result) == {
        ("doc", "ei0", "ei2"): ("EQUAL", False),
        ("doc", "ei0", "ei10"): ("EQUAL", True),
        ("doc", "ei2", "ei10"): ("EQUAL", False),
        ("doc", "ei3", "ei4"): ("EQUAL", False),
        ("doc", "ei3", "ei5"): ("EQUAL", True),
        ("doc", "ei4", "ei5"): ("EQUAL", False),
    }


def test_equality_substitution_expands_both_endpoints_and_transitive_paths():
    result = expand([
        ("doc", "ei1", "ei2", "EQUAL"),
        ("doc", "ei3", "ei4", "EQUAL"),
        ("doc", "ei2", "ei4", "BEFORE"),
        ("doc", "ei4", "ei5", "BEFORE"),
    ])
    assert relations(result) == {
        ("doc", "ei1", "ei2"): ("EQUAL", False),
        ("doc", "ei1", "ei3"): ("BEFORE", True),
        ("doc", "ei1", "ei4"): ("BEFORE", True),
        ("doc", "ei1", "ei5"): ("BEFORE", True),
        ("doc", "ei2", "ei3"): ("BEFORE", True),
        ("doc", "ei2", "ei4"): ("BEFORE", False),
        ("doc", "ei2", "ei5"): ("BEFORE", True),
        ("doc", "ei3", "ei4"): ("EQUAL", False),
        ("doc", "ei3", "ei5"): ("BEFORE", True),
        ("doc", "ei4", "ei5"): ("BEFORE", False),
    }


def test_disconnected_and_unreachable_pairs_are_not_synthetic_vague():
    result = expand([
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "AFTER"),
        ("doc", "ei4", "ei5", "BEFORE"),
        ("abstentions-only", "ei1", "ei2", "NO_PREDS"),
    ])
    assert relations(result) == {
        ("doc", "ei1", "ei2"): ("BEFORE", False),
        ("doc", "ei2", "ei3"): ("AFTER", False),
        ("doc", "ei4", "ei5"): ("BEFORE", False),
    }
    assert set(result.docid) == {"doc"}
    assert "VAGUE" not in result.p_label.tolist()


def test_documents_do_not_share_equality_or_reachability():
    result = expand([
        ("first-doc", "ei1", "ei2", "EQUAL"),
        ("first-doc", "ei2", "ei3", "BEFORE"),
        ("second-doc", "ei1", "ei2", "BEFORE"),
        ("second-doc", "ei2", "ei3", "AFTER"),
    ])
    assert relations(result) == {
        ("first-doc", "ei1", "ei2"): ("EQUAL", False),
        ("first-doc", "ei1", "ei3"): ("BEFORE", True),
        ("first-doc", "ei2", "ei3"): ("BEFORE", False),
        ("second-doc", "ei1", "ei2"): ("BEFORE", False),
        ("second-doc", "ei2", "ei3"): ("AFTER", False),
    }


def test_explicit_vague_wins_but_invalid_labels_do_not_prevent_entailment():
    result = expand([
        ("doc", "ei1", "ei2", " before "),
        ("doc", "ei2", "ei3", "BEFORE"),
        ("doc", "ei1", "ei3", "vague"),
        ("doc", "ei3", "ei4", "BEFORE"),
        ("doc", "ei2", "ei4", "error"),
        ("doc", "ei7", "ei8", "NO_PREDS"),
        ("doc", "ei8", "ei9", None),
        ("doc", "ei9", "ei10", float("nan")),
    ])
    assert relations(result) == {
        ("doc", "ei1", "ei2"): ("BEFORE", False),
        ("doc", "ei1", "ei3"): ("VAGUE", False),
        ("doc", "ei1", "ei4"): ("BEFORE", True),
        ("doc", "ei2", "ei3"): ("BEFORE", False),
        ("doc", "ei2", "ei4"): ("BEFORE", True),
        ("doc", "ei3", "ei4"): ("BEFORE", False),
    }


@pytest.mark.parametrize("inconsistent", [
    [
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "BEFORE"),
        ("doc", "ei1", "ei3", "AFTER"),
        ("doc", "ei3", "ei4", "BEFORE"),
        ("doc", "ei4", "ei5", "BEFORE"),
        ("doc", "ei2", "ei6", "EQUAL"),
        ("doc", "ei6", "ei7", "EQUAL"),
    ],
    [
        ("doc", "ei1", "ei2", "EQUAL"),
        ("doc", "ei2", "ei3", "EQUAL"),
        ("doc", "ei1", "ei3", "BEFORE"),
        ("doc", "ei3", "ei4", "BEFORE"),
        ("doc", "ei4", "ei5", "BEFORE"),
    ],
    [
        ("doc", "ei1", "ei1", "BEFORE"),
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "BEFORE"),
    ],
    [
        ("doc", "ei1", "ei1", "AFTER"),
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "BEFORE"),
    ],
], ids=["directed-cycle", "equality-contradiction", "before-self-loop", "after-self-loop"])
def test_inconsistent_components_keep_explicit_labels_but_do_not_infer(inconsistent):
    healthy = [
        ("doc", "ei10", "ei11", "BEFORE"),
        ("doc", "ei11", "ei12", "BEFORE"),
    ]
    result = expand(inconsistent + healthy)
    expected = {
        (docid, first, second): (label, False)
        for docid, first, second, label in inconsistent + healthy
        if first != second
    }
    expected[("doc", "ei10", "ei12")] = ("BEFORE", True)
    assert relations(result) == expected


def test_equal_self_pair_is_omitted_without_poisoning_a_healthy_component():
    result = expand([
        ("doc", "ei1", "ei1", "EQUAL"),
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "BEFORE"),
    ])
    assert relations(result) == {
        ("doc", "ei1", "ei2"): ("BEFORE", False),
        ("doc", "ei1", "ei3"): ("BEFORE", True),
        ("doc", "ei2", "ei3"): ("BEFORE", False),
    }


def test_canonical_duplicate_rows_are_deduplicated_and_output_order_is_stable():
    rows = [
        ("z", "ei10", "ei2", "AFTER"),
        ("a", "ei2", "ei0", "EQUAL"),
        ("z", "2", "ei010", " earlier "),
        ("a", "ei0", "02", "same"),
        ("z", "ei0", "ei2", "BEFORE"),
    ]
    result = expand(rows)
    pd.testing.assert_frame_equal(result, expand(list(reversed(rows))))
    assert result.values.tolist() == [
        ["a", "ei0", "ei2", "EQUAL", "ei0-ei2", False],
        ["z", "ei0", "ei2", "BEFORE", "ei0-ei2", False],
        ["z", "ei0", "ei10", "BEFORE", "ei0-ei10", True],
        ["z", "ei2", "ei10", "BEFORE", "ei10-ei2", False],
    ]


@pytest.mark.parametrize("first_label,reversed_label", [
    ("BEFORE", "BEFORE"),
    ("BEFORE", "VAGUE"),
    ("EQUAL", "AFTER"),
])
def test_conflicting_canonical_duplicate_labels_fail_instead_of_first_wins(
    first_label, reversed_label,
):
    rows = [
        ("doc", "ei1", "ei2", first_label),
        ("doc", "ei2", "ei1", reversed_label),
    ]
    for ordering in (rows, list(reversed(rows))):
        with pytest.raises(ValueError, match="Conflicting explicit labels.*ei1, ei2"):
            expand(ordering)


def test_input_and_graph_configuration_are_unchanged_and_gold_columns_are_ignored():
    frame = voted([
        ("doc", "ei1", "ei2", "EQUAL"),
        ("doc", "ei2", "ei3", "BEFORE"),
    ])
    frame.index = [8, 3]
    frame["relation"] = ["AFTER", "AFTER"]
    frame["unique_id"] = ["stale", "stale"]
    original = frame.copy(deep=True)
    graph = Graph(
        supported_relation=["AFTER"], relation_key="p_label",
        add_verb_names=True, use_equal=False,
    )
    graph.equal_mapping = {"ei99": "existing-configuration"}
    configuration = deepcopy(graph.__dict__)
    result = graph.generate_implicit_relations(frame)
    assert relations(result) == {
        ("doc", "ei1", "ei2"): ("EQUAL", False),
        ("doc", "ei1", "ei3"): ("BEFORE", True),
        ("doc", "ei2", "ei3"): ("BEFORE", False),
    }
    pd.testing.assert_frame_equal(frame, original)
    assert graph.__dict__ == configuration


def test_configured_relation_key_controls_input_and_output_schema():
    frame = voted([
        ("doc", "ei1", "ei2", "BEFORE"),
        ("doc", "ei2", "ei3", "BEFORE"),
    ]).rename(columns={"p_label": "decision"})
    result = Graph(relation_key="decision").generate_implicit_relations(frame)
    assert result.columns.tolist() == [
        "docid", "eiid1", "eiid2", "decision", "unique_id", "is_inferred"
    ]
    assert result.loc[result.is_inferred, "decision"].tolist() == ["BEFORE"]


@pytest.mark.parametrize("frame", [
    pd.DataFrame(),
    voted([]),
    voted([("doc", "ei1", "ei2", "NO_PREDS")]),
    voted([("doc", "ei1", "ei1", "EQUAL")]),
])
def test_empty_result_always_has_complete_schema_and_boolean_provenance(frame):
    result = Graph(relation_key="p_label").generate_implicit_relations(frame)
    assert result.empty
    assert relations(result) == {}
