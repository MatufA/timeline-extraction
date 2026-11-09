"""Honest full-gold metrics and one-vote-per-trial aggregation."""

import pandas as pd
import pytest

from timeline_extraction.data.postprocessing import majority_vote_decision_new
from timeline_extraction.data.preprocessing import load_data
from timeline_extraction.metrics import calculate_temporal_metrics, summary_results


def gold_rows():
    return pd.DataFrame([
        {"docid": "doc", "unique_id": "ei1-ei2", "label": "BEFORE"},
        {"docid": "doc", "unique_id": "ei2-ei3", "label": "BEFORE"},
    ])


def predictions(rows):
    return pd.DataFrame(rows, columns=["docid", "unique_id", "eiid1", "eiid2", "p_label"])


def test_summary_counts_missing_predictions_in_full_gold_denominator():
    result = summary_results(
        predictions([("doc", "ei1-ei2", "ei1", "ei2", "BEFORE")]), gold_rows(), "local",
    )
    assert result["accuracy"].iloc[0] == 0.5
    assert result["coverage"].iloc[0] == 0.5
    assert result[("BEFORE", "support")].iloc[0] == 2
    assert result[("BEFORE", "precision")].iloc[0] == 1
    assert result[("BEFORE", "recall")].iloc[0] == 0.5
    assert result["micro-f1"].iloc[0] == 0.5


def test_summary_handles_no_predictions_and_single_gold_class():
    result = summary_results(predictions([]), gold_rows(), "local")
    assert result["accuracy"].iloc[0] == 0
    assert result["coverage"].iloc[0] == 0
    assert result["cycles"].iloc[0] == "0 / 1"


def test_summary_invalid_predictions_are_not_fabricated_as_other_labels():
    result = summary_results(
        predictions([("doc", "ei1-ei2", "ei1", "ei2", "invalid")]), gold_rows(), "local",
    )
    assert result["accuracy"].iloc[0] == 0
    assert result["coverage"].iloc[0] == 0
    assert result[("BEFORE", "precision")].iloc[0] == 0


def test_macro_f1_is_unweighted_and_weighted_f1_is_named_separately():
    gold = pd.DataFrame([
        {"docid": "doc", "unique_id": f"ei1-ei{number}", "label": "BEFORE" if number < 5 else "AFTER"}
        for number in range(2, 6)
    ])
    predicted = predictions([
        ("doc", f"ei1-ei{number}", "ei1", f"ei{number}", "BEFORE") for number in range(2, 6)
    ])
    result = summary_results(predicted, gold, "local")
    assert result["macro-f1"].iloc[0] == pytest.approx(3 / 7)
    assert result["weighted-f1"].iloc[0] == pytest.approx(9 / 14)
    assert result["accuracy"].iloc[0] == 0.75


def test_no_vague_evaluation_does_not_drop_wrong_vague_predictions():
    result = calculate_temporal_metrics(["BEFORE", "AFTER", "VAGUE"], ["BEFORE", "VAGUE", "BEFORE"], use_vague=False)
    assert result["accuracy"] == 0.5


def test_invalid_prediction_keeps_alignment_and_denominator():
    result = calculate_temporal_metrics(["BEFORE", "AFTER"], ["invalid", "AFTER"])
    assert result["accuracy"] == 0.5
    with pytest.raises(ValueError, match="counts must match"):
        calculate_temporal_metrics(["BEFORE"], [])


def test_gold_numeric_order_preserves_reversed_relation_and_verbs(tmp_path):
    path = tmp_path / "gold.tsv"
    path.write_text('doc\tlate\tearly\t12\t2\tBEFORE\n' 'doc\tearly\tlate\t2\t12\tAFTER\n')
    gold = load_data(path)
    assert gold.eiid1.tolist() == ["ei2", "ei2"]
    assert gold.eiid2.tolist() == ["ei12", "ei12"]
    assert gold.label.tolist() == ["AFTER", "AFTER"]
    assert gold.relation.tolist() == ["AFTER", "AFTER"]
    assert gold.verb1.tolist() == ["early", "early"]
    assert gold.verb2.tolist() == ["late", "late"]
    result = summary_results(predictions([("doc", "ei12-ei2", "ei2", "ei12", "AFTER")]), gold, "local")
    assert result["accuracy"].iloc[0] == 1


@pytest.mark.parametrize("labels, threshold, expected", [
    (["BEFORE"], 1, ["BEFORE"]),
    (["BEFORE", "BEFORE", "AFTER"], 2, ["BEFORE"]),
    (["BEFORE", "AFTER", "BEFORE", "AFTER"], 3, []),
    (["BEFORE", "BEFORE", "BEFORE", "AFTER", "AFTER", "AFTER", "BEFORE"], 4, ["BEFORE"]),
])
def test_configurable_strict_majority(tmp_path, labels, threshold, expected):
    source, target = tmp_path / "parsed.csv", tmp_path / "voted.csv"
    pd.DataFrame([
        {"docid": "doc", "unique_id": "ei1-ei2", "p_label": label, "iter": trial}
        for trial, label in enumerate(labels)
    ]).to_csv(source, index=False)
    result = majority_vote_decision_new(source, target, threshold)
    assert result.p_label.tolist() == expected


def test_duplicate_rows_and_self_contradicting_trials_cannot_inflate_votes(tmp_path):
    source, target = tmp_path / "parsed.csv", tmp_path / "voted.csv"
    pd.DataFrame([
        {"docid": "doc", "unique_id": "ei1-ei2", "p_label": label, "iter": trial}
        for trial, label in [(0, "BEFORE"), (0, "BEFORE"), (1, "BEFORE"), (1, "AFTER"), (2, "AFTER")]
    ]).to_csv(source, index=False)
    assert majority_vote_decision_new(source, target, 2).empty


def test_empty_parsed_csv_preserves_schema(tmp_path):
    source, target = tmp_path / "parsed.csv", tmp_path / "voted.csv"
    pd.DataFrame(columns=["docid", "unique_id", "p_label", "iter"]).to_csv(source, index=False)
    assert majority_vote_decision_new(source, target).empty
    assert pd.read_csv(target).columns.tolist() == ["docid", "unique_id", "p_label", "iter"]


@pytest.mark.parametrize("equal_pairs", [
    [("ei1", "ei2")],
    [("ei1", "ei3"), ("ei3", "ei2")],
    [("ei3", "ei2"), ("ei1", "ei3")],
])
def test_equality_contradictions_count_even_with_two_edges(equal_pairs):
    from timeline_extraction.metrics import get_n_graph_cycles

    rows = [("doc-with-dashes", e1, e2, "EQUAL") for e1, e2 in equal_pairs]
    rows.append(("doc-with-dashes", "ei1", "ei2", "BEFORE"))
    frame = pd.DataFrame(rows, columns=["docid", "eiid1", "eiid2", "p_label"])
    assert get_n_graph_cycles(frame, "p_label") == {"doc-with-dashes": 1}


def test_equality_contraction_does_not_cross_documents():
    from timeline_extraction.graph import Graph

    frame = pd.DataFrame([
        ("first", "ei1", "ei2", "EQUAL"),
        ("first", "ei1", "ei2", "BEFORE"),
        ("second", "ei1", "ei2", "BEFORE"),
    ], columns=["docid", "eiid1", "eiid2", "p_label"])
    graph = Graph(use_equal=True, supported_relation=("BEFORE", "AFTER", "EQUAL"), relation_key="p_label")
    directed = graph.generate_directed_graph(frame)
    assert ("first-ei1$ei2", "first-ei1$ei2") in directed.edges
    assert ("second-ei1", "second-ei2") in directed.edges
