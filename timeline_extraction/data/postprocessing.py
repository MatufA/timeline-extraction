from itertools import combinations, permutations
import json
import logging
from pathlib import Path
import re
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_matrix

from timeline_extraction.data.preprocessing import Doc, load_data
from timeline_extraction.utils import normalize_event_id

CONVERSE_RELATION = {
    "BEFORE": "AFTER",
    "AFTER": "BEFORE",
    "EQUAL": "EQUAL",
    "VAGUE": "VAGUE",
}

# Ning et al. (2018a) Table 1 deterministic transitivity rules: (r1, r2, r3)
NING_TRANSITIVITY_RULES = (
    ("BEFORE", "BEFORE", "BEFORE"),
    ("BEFORE", "EQUAL", "BEFORE"),
    ("EQUAL", "BEFORE", "BEFORE"),
    ("AFTER", "AFTER", "AFTER"),
    ("AFTER", "EQUAL", "AFTER"),
    ("EQUAL", "AFTER", "AFTER"),
    ("EQUAL", "EQUAL", "EQUAL"),
)

def prepare_df_from_json_response(response: List[dict], doc_obj: Doc, mode: str):
    ei_keys = set(doc_obj.mapping.values())
    data = []

    if isinstance(response, str):
        new_response = []
        for r in response.replace("[", "").replace(",", "").strip().split("\n"):
            try:
                new_response.append(json.loads(r.strip()))
            except json.JSONDecodeError:
                pass
    elif isinstance(response, dict):
        new_response = [response]
    else:
        new_response = response

    for relation in new_response:
        e1, e2 = (
            (f'ei{relation["event1"]}', f'ei{relation["event2"]}')
            if mode != "pair"
            else (relation["event1"], relation["event2"])
        )
        relation = relation["relation"]
        if e1 not in ei_keys or e2 not in ei_keys:
            unique_id = "UNDEFINED"
        else:
            unique_id = "-".join(sorted([e1, e2]))

        if relation == "after":
            e1, e2 = e2, e1
            label = "after"
            relation = "before"
        else:
            label = relation

        data.append(
            {
                "docid": doc_obj.docid,
                "eiid1": e1.strip(),
                "eiid2": e2.strip(),
                "relation": relation,
                "unique_id": unique_id,
                "p_label": label.upper(),
                "mode": mode,
            }
        )
    return pd.DataFrame(data)


def prepare_df_from_response(response: str, doc_obj: Doc, mode: str):
    supported_relations = ("before", "after", "equal", "vague")
    ei_keys = set(doc_obj.mapping.values())
    data = []
    for line in response.strip().splitlines():
        e1e2_opt = re.search(r"e.*", line)
        if e1e2_opt is None:
            continue

        e1e2 = e1e2_opt.group(0).strip()

        regex_parsers = [
            r"(ei\d+):\w+\s+(\w+)\s+(ei\d+).*",
            r"(ei\d+)\s+(\w+)\s+(ei\d+).*",
            r"(ei\d+)\s+\w+\s+(\w+)\s+(ei\d+).*",
            r"(ei\d+):(\w+)\s+(ei\d+).*",
            r"(ei\[\d+\])\s+(\w+)\s+(ei\[\d+\]).*",
            r"(ei\d+)\s+\[\w+\]\s+(ei\d+):\s+(\w+)",
            r"(ei\d+)\s+\[(\w+)\]\s+(ei\d+)",
            r"(ei\d+):\s*\w+\s+(\w+)\s+(ei\d+).*",
        ]

        e1_relation_e2 = None
        for regex_parser in regex_parsers:
            e1_relation_e2 = re.search(regex_parser, e1e2.lower())
            if e1_relation_e2 is not None:
                break

        if e1_relation_e2 is None:
            logging.error(f'unable to parse line: "{line}", ignored line')
            continue

        e1, relation, e2 = e1_relation_e2.groups()
        e1 = e1.strip()
        e2 = e2.strip()
        relation = relation.strip()

        if relation not in supported_relations:
            if e2 in supported_relations:
                e2, relation = relation, e2
            else:
                continue

        if e1 not in ei_keys or e2 not in ei_keys:
            unique_id = "UNDEFINED"
            # verb1 = doc_obj.mapping.get(e1)
            # verb2 = doc_obj.mapping.get(e2)
        else:
            unique_id = "-".join(sorted([e1, e2]))
            # verb1 = doc_obj.mapping[e1]
            # verb2 = doc_obj.mapping[e2]

        if relation == "after":
            # verb1, verb2 = verb2, verb1
            e1, e2 = e2, e1
            label = "after"
            relation = "before"
        else:
            label = relation

        data.append(
            {
                "docid": doc_obj.docid,
                # 'verb1': verb1,
                # 'verb2': verb2,
                "eiid1": e1.strip(),
                "eiid2": e2.strip(),
                "relation": relation,
                "unique_id": unique_id,
                "p_label": label.upper(),
                "mode": mode,
            }
        )
    return pd.DataFrame(data)


def transform_to_before(df, doc_id):
    doc_events = df[df.docid == doc_id].copy()
    doc_events = doc_events.apply(
        lambda row: (
            (row.verb2, row.verb1, row.eiid2, row.eiid1, "BEFORE")
            if row.relation == "AFTER"
            else (row.verb1, row.verb2, row.eiid1, row.eiid2, row.relation)
        ),
        axis="columns",
        result_type="expand",
    )
    return doc_events


def _extract_pair_probabilities(row, labels: tuple[str, ...]) -> np.ndarray | None:
    raw = row.get("raw_response")
    if isinstance(raw, str) and raw.lstrip().startswith("{"):
        try:
            payload = json.loads(raw)
            probs = payload.get("probabilities") if isinstance(payload, dict) else None
            if isinstance(probs, dict) and probs:
                vec = np.array(
                    [float(probs.get(label.lower(), probs.get(label, 0.0))) for label in labels],
                    dtype=float,
                )
                total = vec.sum()
                if total > 0:
                    return vec / total
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if "probs" in row and pd.notna(row["probs"]):
        pred = str(row.get("p_label", row.get("relation", ""))).strip().upper()
        if pred in labels:
            p_val = float(np.clip(float(row["probs"]), 1e-4, 1.0 - 1e-4))
            rem = (1.0 - p_val) / max(len(labels) - 1, 1)
            vec = np.full(len(labels), rem, dtype=float)
            vec[labels.index(pred)] = p_val
            return vec
    return None


def _find_contradiction_cut(pairs, chosen_labels, var_idx):
    """Find a minimal cycle or equality-contraction contradiction as ILP variable indices."""
    eq_graph = nx.Graph()
    for (u, v), rel in zip(pairs, chosen_labels):
        eq_graph.add_nodes_from((u, v))
        if rel == "EQUAL":
            eq_graph.add_edge(u, v)

    rep = {}
    for comp in nx.connected_components(eq_graph):
        r = min(comp)
        for node in comp:
            rep[node] = r

    def eq_path_vars(a: int, b: int) -> list[int]:
        if a == b:
            return []
        path = nx.shortest_path(eq_graph, a, b)
        return [var_idx(path[i], path[i + 1], "EQUAL") for i in range(len(path) - 1)]

    di_graph = nx.DiGraph()
    edge_meta = {}
    for (u, v), rel in zip(pairs, chosen_labels):
        if rel not in {"BEFORE", "AFTER"}:
            continue
        src, dst = (u, v) if rel == "BEFORE" else (v, u)
        r_src, r_dst = rep[src], rep[dst]
        v_idx = var_idx(u, v, rel)
        if r_src == r_dst:
            return list(dict.fromkeys([v_idx, *eq_path_vars(src, dst)]))
        di_graph.add_edge(r_src, r_dst)
        edge_meta[(r_src, r_dst)] = (src, dst, v_idx)

    try:
        cycle = nx.find_cycle(di_graph, orientation="original")
    except nx.NetworkXNoCycle:
        return None

    cut_vars = []
    k = len(cycle)
    for idx in range(k):
        r_u, r_v, _ = cycle[idx]
        src_u, dst_v, v_idx = edge_meta[(r_u, r_v)]
        cut_vars.append(v_idx)
        next_r_u, next_r_v, _ = cycle[(idx + 1) % k]
        next_src_u, _, _ = edge_meta[(next_r_u, next_r_v)]
        if dst_v != next_src_u:
            cut_vars.extend(eq_path_vars(dst_v, next_src_u))
    return list(dict.fromkeys(cut_vars))


def transitive_constraints_decision_ilp(
    df: pd.DataFrame,
    *,
    use_vague: bool | None = None,
    label_col: str = "p_label",
) -> pd.DataFrame:
    """Global temporal consistency via Ning et al. (2018a) Integer Linear Programming.

    Maximizes sum_{(i,j)} sum_{r} P(r | i, j) * I_{ij}^r subject to:
      1. Uniqueness (simplex): sum_r I_{ij}^r = 1 for all pairs (i, j)
      2. Symmetry (converse): I_{ji}^r = I_{ij}^{converse(r)}
      3. Transitivity: I_{ij}^{r1} + I_{jk}^{r2} - I_{ik}^{r3} <= 1 for all triples
         and generalized cycle-elimination cuts on sparse graphs.
    """
    if df.empty:
        return df.copy()
    if use_vague is None:
        has_vague_label = label_col in df and (df[label_col].astype(str).str.upper() == "VAGUE").any()
        has_vague_prob = "raw_response" in df and df["raw_response"].astype(str).str.contains('"vague"', case=False, na=False).any()
        use_vague = bool(has_vague_label or has_vague_prob)
    labels = ("BEFORE", "AFTER", "EQUAL", "VAGUE") if use_vague else ("BEFORE", "AFTER", "EQUAL")
    label_idx = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)

    out_frames = []
    for docid, group in df.groupby("docid", sort=False):
        pair_records = {}
        pair_probs = {}
        for unique_id, pair_group in group.groupby("unique_id", sort=False):
            first_row = pair_group.iloc[0]
            u = int(normalize_event_id(str(first_row["eiid1"]))[2:])
            v = int(normalize_event_id(str(first_row["eiid2"]))[2:])
            swapped = u > v
            if swapped:
                u, v = v, u
            pair_key = (u, v)
            vecs = []
            for _, row in pair_group.iterrows():
                vec = _extract_pair_probabilities(row, labels)
                if vec is not None:
                    if swapped:
                        vec = vec.copy()
                        vec[label_idx["BEFORE"]], vec[label_idx["AFTER"]] = (
                            vec[label_idx["AFTER"]],
                            vec[label_idx["BEFORE"]],
                        )
                    vecs.append(vec)
            if vecs:
                avg_vec = np.mean(vecs, axis=0)
            else:
                counts = np.zeros(n_labels, dtype=float)
                for raw_label in pair_group[label_col].astype(str).str.upper():
                    oriented = CONVERSE_RELATION.get(raw_label, raw_label) if swapped else raw_label
                    if oriented in label_idx:
                        counts[label_idx[oriented]] += 1.0
                total = counts.sum()
                avg_vec = counts / total if total > 0 else np.full(n_labels, 1.0 / n_labels)
            # Tiny deterministic tie-break toward highest probability option
            avg_vec = avg_vec + np.arange(n_labels, 0, -1) * 1e-9
            pair_probs[pair_key] = avg_vec
            rep_row = first_row.copy()
            if swapped:
                rep_row["eiid1"], rep_row["eiid2"] = f"ei{u}", f"ei{v}"
            pair_records[pair_key] = rep_row

        pairs = sorted(pair_probs)
        n_pairs = len(pairs)
        if n_pairs == 0:
            continue
        pair_map = {pair: idx for idx, pair in enumerate(pairs)}
        prob_matrix = np.vstack([pair_probs[pair] for pair in pairs])
        n_vars = n_pairs * n_labels

        def var_idx(u: int, v: int, rel: str) -> int:
            if u < v:
                return pair_map[(u, v)] * n_labels + label_idx[rel]
            return pair_map[(v, u)] * n_labels + label_idx[CONVERSE_RELATION[rel]]

        eq_rows = np.repeat(np.arange(n_pairs), n_labels)
        eq_cols = np.arange(n_vars)
        a_eq = csc_matrix((np.ones(n_vars), (eq_rows, eq_cols)), shape=(n_pairs, n_vars))

        nodes = sorted({node for pair in pairs for node in pair})
        ub_rows, ub_cols, ub_vals = [], [], []
        row_count = 0
        seen_triples = set()
        for u, v, w in combinations(nodes, 3):
            if (u, v) in pair_map and (v, w) in pair_map and (u, w) in pair_map:
                for i, j, k in permutations((u, v, w), 3):
                    for r1, r2, r3 in NING_TRANSITIVITY_RULES:
                        if r1 not in label_idx or r2 not in label_idx or r3 not in label_idx:
                            continue
                        v1 = var_idx(i, j, r1)
                        v2 = var_idx(j, k, r2)
                        v3 = var_idx(i, k, r3)
                        key = (min(v1, v2), max(v1, v2), v3)
                        if key in seen_triples:
                            continue
                        seen_triples.add(key)
                        ub_rows.extend((row_count, row_count, row_count))
                        ub_cols.extend((v1, v2, v3))
                        ub_vals.extend((1.0, 1.0, -1.0))
                        row_count += 1

        extra_cuts = []
        while True:
            constraints = [LinearConstraint(a_eq, np.ones(n_pairs), np.ones(n_pairs))]
            total_rows = row_count + len(extra_cuts)
            if total_rows > 0:
                all_r = list(ub_rows)
                all_c = list(ub_cols)
                all_v = list(ub_vals)
                all_rhs = [1.0] * row_count
                for c_idx, cut in enumerate(extra_cuts, start=row_count):
                    for v_col in cut:
                        all_r.append(c_idx)
                        all_c.append(v_col)
                        all_v.append(1.0)
                    all_rhs.append(float(len(cut) - 1))
                a_ub = csc_matrix((all_v, (all_r, all_c)), shape=(total_rows, n_vars))
                constraints.append(LinearConstraint(a_ub, -np.inf, np.array(all_rhs)))

            res = milp(
                c=-prob_matrix.ravel(),
                integrality=np.ones(n_vars),
                bounds=(np.zeros(n_vars), np.ones(n_vars)),
                constraints=constraints,
            )
            if res.x is None:
                chosen_indices = prob_matrix.argmax(axis=1)
                chosen_labels = [labels[idx] for idx in chosen_indices]
                break
            chosen_indices = np.round(res.x).reshape(n_pairs, n_labels).argmax(axis=1)
            chosen_labels = [labels[idx] for idx in chosen_indices]
            cut = _find_contradiction_cut(pairs, chosen_labels, var_idx)
            if not cut or cut in extra_cuts:
                break
            extra_cuts.append(cut)

        for pair_key, chosen_label in zip(pairs, chosen_labels):
            rec = pair_records[pair_key].copy()
            rec[label_col] = chosen_label
            out_frames.append(rec)

    if not out_frames:
        return df.iloc[:0].copy()
    return pd.DataFrame(out_frames, columns=df.columns)


def majority_vote_decision_new(
    parsed_response_path: Path, results_path: Path, min_votes: int = 3
):
    """
    Aggregate parsed trial responses into document predictions.

    When responses carry calibrated probability distributions (such as Gemini Decision 1),
    solves the Ning et al. (2018a) Integer Linear Programming (ILP) transitive constraints
    optimization across each document instead of independent majority voting.
    """
    df = pd.read_csv(parsed_response_path)
    votes = df[df["p_label"].isin(["BEFORE", "AFTER", "EQUAL", "VAGUE"])].copy()
    if "iter" in votes:
        trial_keys = ["docid", "unique_id", "iter"]
        votes = votes.drop_duplicates(trial_keys + ["p_label"])
        # A trial that contradicts itself cannot cast multiple votes for a pair.
        conflicts = votes.groupby(trial_keys)["p_label"].transform("nunique") > 1
        votes = votes[~conflicts]

    has_calibrated_probabilities = (
        not votes.empty
        and "raw_response" in votes.columns
        and votes["raw_response"]
        .astype(str)
        .str.contains('"probabilities"', na=False)
        .any()
    )
    if has_calibrated_probabilities:
        filtered_df = transitive_constraints_decision_ilp(votes, label_col="p_label")
        filtered_df.to_csv(results_path, index=False)
        return filtered_df

    vote_counts = (
        votes.groupby(["docid", "unique_id", "p_label"])
        .size()
        .reset_index(name="vote_count")
    )
    qualified_votes = vote_counts[vote_counts["vote_count"] >= min_votes]
    if qualified_votes.empty:
        filtered_df = df.iloc[:0].copy()
    else:
        majority_labels = qualified_votes.loc[
            qualified_votes.groupby(["docid", "unique_id"])["vote_count"].idxmax()
        ]
        filtered_df = pd.merge(
            votes.drop_duplicates(subset=["docid", "unique_id", "p_label"]),
            majority_labels[["docid", "unique_id", "p_label"]],
            on=["docid", "unique_id", "p_label"],
            how="inner",
        )

    # Save the filtered results
    filtered_df.to_csv(results_path, index=False)

    return filtered_df

def majority_vote_decision(
    parsed_response_path: Path, results_path: Path, min_votes: int = 3
):
    predicted_df = pd.read_csv(parsed_response_path)

    predicted_df["score"] = 1
    predicted_sum_df = (
        predicted_df.groupby(["docid", "unique_id", "relation"])["score"]
        .sum()
        .reset_index()
    )
    predicted_sum_df = predicted_sum_df[predicted_sum_df["score"] >= min_votes]

    agg_preds = []
    for (docid, unique_id), group in predicted_sum_df.groupby(["docid", "unique_id"]):
        max_score = group.score.max()
        relations = group[group.score == max_score]["relation"].to_list()

        if len(relations) > 1:
            agg_preds.append(
                {
                    "docid": docid,
                    "unique_id": unique_id,
                    "relation": "VAGUE",
                    "max_score": max_score,
                    "conflict_rel": ",".join(relations),
                }
            )
        else:
            agg_preds.append(
                {
                    "docid": docid,
                    "unique_id": unique_id,
                    "relation": relations[0].upper(),
                    "max_score": max_score,
                }
            )

    agg_pred_df = pd.DataFrame(agg_preds)
    predicted_df.relation = predicted_df.relation.str.upper()

    final_df = pd.merge(
        predicted_df,
        agg_pred_df,
        on=["docid", "unique_id"],
        suffixes=(None, "_selected"),
        how="left",
    )
    final_df["min_vote"] = min_votes

    final_df.to_csv(results_path, index=False)
    return final_df


if __name__ == "__main__":
    from timeline_extraction.pipeline import generate_suffix_name

    DATA_PATH = Path("./data")
    MATRES_DATA_PATH = DATA_PATH / "MATRES"
    PLATINUM_RAW = MATRES_DATA_PATH / "raw" / "TBAQ-cleaned" / "te3-platinum"
    gold_data_path = MATRES_DATA_PATH / "platinum.txt"
    TRC_RAW_PATH = DATA_PATH / "TRC"
    GRAPH_BASE_PATH = DATA_PATH / "graph_exploration"

    gold_df = load_data(gold_data_path)
    gold_subcolumn_df = gold_df[["docid", "eiid1", "eiid2", "label"]].copy()
    gold_subcolumn_df.columns = ["doc_id", "event1", "event2", "relation"]
    gold_relation_dict = gold_subcolumn_df.to_dict(orient="records")

    with (GRAPH_BASE_PATH / "platinum_graph").open("w") as gold_file:
        gold_file.write(json.dumps(gold_relation_dict))

    data_name = "te3-platinum"

    # modes = 'pair'
    mode = "multi"
    # mode = 'comb'

    use_vague = True

    methods = ["zero-shot", "few-shot"]
    model_names = ["gpt-4o-mini"]

    suffix_path = "completion"

    for model_name in model_names:
        for method in methods:
            suffix_name = generate_suffix_name(
                model_name, method, suffix_path, use_vague
            )
            parsed_response_path = (
                TRC_RAW_PATH
                / "parsed_responses"
                / method
                / f"{mode}-{data_name}-results-{suffix_name}.csv"
            )
            model_response_df = pd.DataFrame(parsed_response_path)
