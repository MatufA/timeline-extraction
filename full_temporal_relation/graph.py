from functools import reduce
import networkx as nx
import pandas as pd

from itertools import chain, product, starmap
from functools import partial


class Graph:
    def __init__(self, supported_relation=('AFTER', 'BEFORE'), relation_key: str = 'relation'):
        self.supported_relation = supported_relation
        self.relation_key = relation_key

    def create_edges(self, df):
        edges_df = pd.DataFrame()
        edges_df[['source', 'target']] = df[['docid', 'eiid1', 'eiid2', self.relation_key]] \
            .apply(lambda row: (f"{row.docid}-{row['eiid2']}", f"{row.docid}-{row['eiid1']}")
        if row[self.relation_key] == 'AFTER' else (f"{row.docid}-{row['eiid1']}", f"{row.docid}-{row['eiid2']}"),
                   axis='columns', result_type="expand")
        return edges_df

    def create_nodes(self, df):
        nodes_raw_df = df.apply(
            lambda row: [dict(docid=row.docid, verb=row.verb1, eiid=f"{row.docid}-{row.eiid1}", eid=row.eiid1),
                         dict(docid=row.docid, verb=row.verb2, eiid=f"{row.docid}-{row.eiid2}", eid=row.eiid2)],
            axis='columns').to_list()
        nodes_flatten = reduce(lambda x, y: x + y, nodes_raw_df)
        return pd.DataFrame(nodes_flatten).drop_duplicates(ignore_index=True)

    def generate_directed_graph(self, df):
        df = df.loc[df[self.relation_key].isin(self.supported_relation)]
        edges = self.create_edges(df)
        nodes = self.create_nodes(df)

        # G = nx.DiGraph()
        G = nx.from_pandas_edgelist(edges, create_using=nx.DiGraph)
        nx.set_node_attributes(G, dict(zip(nodes.eiid, nodes.docid)), 'docid')
        nx.set_node_attributes(G, dict(zip(nodes.eiid, nodes.verb)), 'verb')
        nx.set_node_attributes(G, dict(zip(nodes.eiid, nodes.eid)), 'eid')
        return G

    def generate_subgraph_by_docid(self, graph, doc_id):
        sub_graph = []
        for (n, data) in graph.nodes(data=True):
            if data['docid'] == doc_id:
                sub_graph.append(n)

        return graph.subgraph(sub_graph).copy()

    def find_cycles(self, df):
        cycles = {}
        idx = 0
        for _, group in df.groupby('docid'):
            if group[self.relation_key].isin(self.supported_relation).count() < 3:
                continue
            try:
                sub_graph = self.generate_directed_graph(df=group)
                cycle = nx.find_cycle(sub_graph, orientation='original')
                docid = cycle[0][0].split('-')[0]
                if docid in cycles:
                    cycles[docid].append(cycle)
                else:
                    cycles[docid] = [cycle]
            except nx.NetworkXNoCycle:
                idx += 1
            except ValueError:
                print(group)

        return cycles


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
