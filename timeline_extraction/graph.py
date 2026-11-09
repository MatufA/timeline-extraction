from functools import reduce
import networkx as nx
import pandas as pd
import logging

from itertools import chain, combinations, product, starmap
from functools import partial

from timeline_extraction.utils import (
    create_unique_id,
    normalize_event_id,
    normalize_relation,
    validate_temporal_relation,
)


class Graph:
    def __init__(
        self,
        supported_relation=("AFTER", "BEFORE"),
        relation_key: str = "relation",
        add_verb_names: bool = False,
        use_equal: bool = False,
    ):
        self.supported_relation = supported_relation
        self.relation_key = relation_key
        self.add_verb_names = add_verb_names
        self.use_equal = use_equal
        self.equal_mapping = {}

    def create_edges(self, df):
        edges_df = pd.DataFrame()
        edges_df[["source", "target"]] = df[
            ["docid", "eiid1", "eiid2", self.relation_key]
        ].apply(
            lambda row: (
                (f"{row.docid}-{row['eiid2']}", f"{row.docid}-{row['eiid1']}")
                if row[self.relation_key] == "AFTER"
                else (f"{row.docid}-{row['eiid1']}", f"{row.docid}-{row['eiid2']}")
            ),
            axis="columns",
            result_type="expand",
        )
        return edges_df

    def create_nodes(self, df):
        if self.add_verb_names:
            nodes_raw_df = df.apply(
                lambda row: [
                    dict(
                        docid=row.docid,
                        verb=row.verb1,
                        eiid=f"{row.docid}-{row.eiid1}",
                        eid=row.eiid1,
                    ),
                    dict(
                        docid=row.docid,
                        verb=row.verb2,
                        eiid=f"{row.docid}-{row.eiid2}",
                        eid=row.eiid2,
                    ),
                ],
                axis="columns",
            ).to_list()
        else:
            nodes_raw_df = df.apply(
                lambda row: [
                    dict(
                        docid=row.docid, eiid=f"{row.docid}-{row.eiid1}", eid=row.eiid1
                    ),
                    dict(
                        docid=row.docid, eiid=f"{row.docid}-{row.eiid2}", eid=row.eiid2
                    ),
                ],
                axis="columns",
            ).to_list()
        nodes_flatten = reduce(lambda x, y: x + y, nodes_raw_df)
        return pd.DataFrame(nodes_flatten).drop_duplicates(ignore_index=True)

    def _fit_df_to_equal(self, df):
        """Contract complete equality components independently within each document."""
        self.equal_mapping = {}
        fitted = df.copy()
        for docid, group in df.groupby("docid"):
            equal_df = group.loc[group[self.relation_key] == "EQUAL", ["eiid1", "eiid2"]]
            equality_graph = nx.Graph()
            equality_graph.add_edges_from(equal_df.itertuples(index=False, name=None))
            mapping = {}
            for component in nx.connected_components(equality_graph):
                key = "$".join(sorted(component))
                mapping.update(dict.fromkeys(component, key))
            self.equal_mapping.update(mapping)
            mask = fitted["docid"] == docid
            for column in ("eiid1", "eiid2"):
                fitted.loc[mask, column] = fitted.loc[mask, column].replace(mapping)
        return fitted

    def generate_directed_graph(self, df):
        if self.use_equal:
            if "EQUAL" in self.supported_relation:
                df = self._fit_df_to_equal(df)
                df = df[df[self.relation_key] != "EQUAL"]
            else:
                logging.error("'EQUAL' not in supported relation, ignoring it.")

        df = df.loc[df[self.relation_key].isin(self.supported_relation)]

        if df.empty:
            logging.warning(
                "there is no supported relation in datafarme ot it empty, return empty graph"
            )
            return nx.DiGraph()

        edges = self.create_edges(df)
        nodes = self.create_nodes(df)

        G = nx.from_pandas_edgelist(edges, create_using=nx.DiGraph)
        nx.set_node_attributes(G, dict(zip(nodes.eiid, nodes.docid)), "docid")
        nx.set_node_attributes(G, dict(zip(nodes.eiid, nodes.eid)), "eid")

        if self.add_verb_names:
            nx.set_node_attributes(G, dict(zip(nodes.eiid, nodes.verb)), "verb")

        return G

    def generate_subgraph_by_docid(self, graph, doc_id):
        sub_graph = []
        for n, data in graph.nodes(data=True):
            if data["docid"] == doc_id:
                sub_graph.append(n)

        return graph.subgraph(sub_graph).copy()

    def find_cycles(self, df):
        cycles = {}
        idx = 0
        for docid, group in df.groupby("docid"):
            if not group[self.relation_key].isin(self.supported_relation).any():
                continue
            try:
                sub_graph = self.generate_directed_graph(df=group)
                cycle = nx.find_cycle(sub_graph, orientation="original")
                if docid in cycles:
                    cycles[docid].append(cycle)
                else:
                    cycles[docid] = [cycle]
            except nx.NetworkXNoCycle:
                idx += 1
            except ValueError:
                print(group)

        return cycles

    def find_simple_cycles(self, df):
        cycles_list = {}
        idx = 0
        for _, group in df.groupby("docid"):
            if group[self.relation_key].isin(self.supported_relation).count() < 3:
                continue
            try:
                sub_graph = self.generate_directed_graph(df=group)
                cycles = nx.simple_cycles(sub_graph)
                for cycle in cycles:
                    docid = cycle[0].split("-")[0]
                    if docid in cycles:
                        cycles_list[docid].append(cycle)
                    else:
                        cycles_list[docid] = [cycle]
            except nx.NetworkXNoCycle:
                idx += 1
            except ValueError:
                print(group)

        return cycles

    def generate_implicit_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete voted predictions without consulting gold or changing this graph.

        Explicit labels take precedence over inferred labels. Conflicting duplicate
        pairs raise ValueError after numeric canonicalization. Equality is contracted
        per document; a directed cycle (including a contraction-induced self-loop)
        disables all inference in its weakly connected component. Self-pairs are
        never returned, but explicit temporal self-loops still block inference.
        """
        columns = [
            "docid", "eiid1", "eiid2", self.relation_key, "unique_id", "is_inferred"
        ]
        output = []
        if df.empty:
            return pd.DataFrame(columns=columns).astype({"is_inferred": bool})

        for docid, group in df.groupby("docid", sort=True):
            relations = {}
            for first, second, label in group[
                ["eiid1", "eiid2", self.relation_key]
            ].itertuples(index=False, name=None):
                if not isinstance(label, str):
                    continue
                label = normalize_relation(label)
                if not validate_temporal_relation(label):
                    continue
                first = int(normalize_event_id(str(first))[2:])
                second = int(normalize_event_id(str(second))[2:])
                if first > second:
                    first, second = second, first
                    label = {"BEFORE": "AFTER", "AFTER": "BEFORE"}.get(label, label)
                pair = (first, second)
                if pair in relations and relations[pair][0] != label:
                    raise ValueError(
                        f"Conflicting explicit labels for {docid}: ei{first}, ei{second}"
                    )
                relations[pair] = (label, False)

            equality = nx.Graph()
            for (first, second), (label, _) in relations.items():
                equality.add_nodes_from((first, second))
                if label == "EQUAL":
                    equality.add_edge(first, second)
            members = {}
            representatives = {}
            for component in nx.connected_components(equality):
                ordered = sorted(component)
                representative = ordered[0]
                members[representative] = ordered
                representatives.update(dict.fromkeys(component, representative))

            before = nx.DiGraph()
            before.add_nodes_from(members)
            for (first, second), (label, _) in relations.items():
                if label == "AFTER":
                    first, second = second, first
                if label in {"BEFORE", "AFTER"}:
                    before.add_edge(representatives[first], representatives[second])

            for component in nx.weakly_connected_components(before):
                directed = before.subgraph(component)
                if not nx.is_directed_acyclic_graph(directed):
                    continue
                for representative in component:
                    for pair in combinations(members[representative], 2):
                        relations.setdefault(pair, ("EQUAL", True))
                # Compute reachability once, then expand contracted equality nodes.
                closure = nx.transitive_closure_dag(directed)
                for source, target in closure.edges:
                    for first, second in product(members[source], members[target]):
                        if first < second:
                            pair, label = (first, second), "BEFORE"
                        else:
                            pair, label = (second, first), "AFTER"
                        relations.setdefault(pair, (label, True))

            for (first, second), (label, inferred) in sorted(relations.items()):
                if first == second:
                    continue
                eiid1, eiid2 = f"ei{first}", f"ei{second}"
                output.append((
                    docid, eiid1, eiid2, label,
                    create_unique_id(eiid1, eiid2), inferred,
                ))

        return pd.DataFrame(output, columns=columns).astype({"is_inferred": bool})


def create_simple_graph(graph):
    # find connected components
    conn_components = nx.connected_components(graph.to_undirected())

    simple_edges = []
    for nodes in conn_components:
        # create copy of DiGraph
        sub_shallow = graph.subgraph(nodes).copy()

        if len(sub_shallow.edges) <= 2:
            simple_edges.extend([edge for edge in sub_shallow.edges])
            continue

        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(sub_shallow.nodes(data=True))
        new_graph.add_edges_from(sub_shallow.edges)

        # find simple paths
        chaini = chain.from_iterable
        roots = (v for v, d in new_graph.in_degree() if d == 0)
        leaves = (v for v, d in new_graph.out_degree() if d == 0)

        all_paths = partial(nx.all_simple_paths, new_graph)
        all_simple_paths = chaini(starmap(all_paths, product(roots, leaves)))

        # remove redundant paths
        sorted_path = sorted(all_simple_paths, key=lambda x: -len(x))
        max_path = set(sorted_path[0])

        simple_edges.append(sorted_path[0])
        for path in sorted_path[1:]:
            if len(max_path | set(path)) > len(max_path):
                max_path.update(path)
                simple_edges.append(path)

    simple_graph = nx.DiGraph()
    simple_graph.add_nodes_from(graph.nodes(data=True))

    for path in simple_edges:
        nx.add_path(simple_graph, path)

    return simple_graph, simple_edges


def break_cycles_by_confidence(cycles: list, df: pd.DataFrame) -> pd.DataFrame:
    for cycle in cycles:
        cycle_data = [
            (e1.split("-")[1], e2.split("-")[1], relation) for e1, e2, relation in cycle
        ]
        cycle_df = pd.DataFrame(cycle_data, columns=["eiid1", "eiid2", "relation"])
        eiid1_eiid2 = list(zip(cycle_df["eiid1"], cycle_df["eiid2"]))
        cycle_df["unique_id"] = [
            "-".join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2
        ]
        cycle_scores = pd.merge(df[["unique_id", "probs"]], cycle_df, on=["unique_id"])
        edge_to_drop = cycle_scores.iloc[cycle_scores.probs.argmin()]
        unique_id = "-".join(sorted([edge_to_drop["eiid1"], edge_to_drop["eiid2"]]))

        df = df.drop(df[df.unique_id == unique_id].index)
    return df


if __name__ == "__main__":
    from pathlib import Path

    TRC_GLOBAL_VARIABLES = {
        "LABELS": ["BEFORE", "AFTER", "EQUAL", "VAGUE"],
        "LABELS_NO_VAGUE": ["BEFORE", "AFTER", "EQUAL"],
        "LABELS_IDS": [0, 1, 2, 3],
    }
    id2label = dict(
        zip(TRC_GLOBAL_VARIABLES["LABELS_IDS"], TRC_GLOBAL_VARIABLES["LABELS"])
    )

    DATA_PATH = Path("./data")
    MATRES_DATA_PATH = DATA_PATH / "MATRES"
    PLATINUM_RAW = MATRES_DATA_PATH / "raw" / "TBAQ-cleaned" / "te3-platinum"
    gold_data_path = MATRES_DATA_PATH / "platinum.txt"

    graph_example = DATA_PATH / "graph_exploration" / "te3-platinum-baseline-graph"

    baseline_preds_path = DATA_PATH / "narrativetime_a1_test_predict_with_probs.csv"
    baseline_no_cycles_path = (
        DATA_PATH / "narrativetime_a1_test_predict_baseline_no_cycles_by_probs.csv"
    )

    # gold_df = load_data(gold_data_path)
    gold_df = pd.read_csv(baseline_preds_path)
    # gold_df = gold_df.rename(columns={'doc_id': 'docid'})
    # gold_df['relation'] = gold_df.predictions.replace(id2label)
    # eiid1_eiid2 = list(zip(gold_df['eiid1'], gold_df['eiid2']))
    # gold_df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

    graph = Graph(use_equal=True, supported_relation=["AFTER", "BEFORE"])

    print(gold_df.shape)

    data = []
    for docid, cycles in graph.find_cycles(gold_df).items():
        for cycle in cycles:
            data.extend(
                [docid, eiid1.split("-")[1], eiid2.split("-")[1]]
                for eiid1, eiid2, relation in cycle
            )

    df_cycles = pd.DataFrame(data, columns=["docid", "eiid1", "eiid2"])
    eiid1_eiid2 = list(zip(df_cycles["eiid1"], df_cycles["eiid2"]))
    df_cycles["unique_id"] = [
        "-".join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2
    ]

    # cycle_only_df = pd.merge(gold_df, df_cycles[['docid', 'unique_id']], on=['docid', 'unique_id'], how='inner').drop_duplicates()
    # cycle_only_df.to_csv(DATA_PATH / 'comb_te3-platinum_minimal_context_predictions_cycles_only.csv', index=False)
    all_cycles = {}
    # gold_df = gold_df[gold_df.docid == 'nyt_20130321_china_pollution']
    while sum(len(v) for v in graph.find_cycles(gold_df).values()) > 0:
        cycles = {k: v for k, v in graph.find_cycles(gold_df).items()}
        # print(f'cycles: {cycles}')
        no_cycle_data = []

        for docid, group in gold_df.groupby("docid"):
            if docid in cycles:
                doc_cycles = cycles[docid]

                if docid in all_cycles:
                    all_cycles[docid].extend(doc_cycles)
                else:
                    all_cycles[docid] = doc_cycles

                no_cycle_data.append(
                    break_cycles_by_confidence(cycles=doc_cycles, df=group)
                )
            else:
                no_cycle_data.append(group)

            # doc_graph = graph.generate_directed_graph(group)
            # doc__simple_graph, _ = create_simple_graph(doc_graph)
            # plt_graph = draw_directed_graph(doc__simple_graph, label_name='eid', cycles_only=False)
            # plt_graph.savefig(graph_example / f'{docid}.png')
            # plt_graph.clf()
        gold_df = pd.concat(no_cycle_data)
        print(gold_df.shape)
        # cycles2 = {k: v for k, v in graph.find_cycles(gold_df).items()}
        print(f"cycles2: {sum(len(v) for v in graph.find_cycles(gold_df).values())}")

    # import json
    # output = Path('/home/adiel/full-temporal-relation/data/all_cycles_matres_by_baseline.json')
    # output.write_text(json.dumps(all_cycles, indent=2))

    gold_df.to_csv(baseline_no_cycles_path, index=False)

    print("Done!")
