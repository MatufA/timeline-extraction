import networkx as nx
import matplotlib.pyplot as plt


def draw_directed_graph(graph, title=None, label_name=None, cycles_only: bool = False):
    plt.rcParams["figure.figsize"] = (15, 10)

    labels = nx.get_node_attributes(graph, label_name)

    try:
        cycles = nx.find_cycle(graph, orientation="original")
        pos = nx.spring_layout(graph, seed=47, k=1, iterations=50)
    except nx.NetworkXNoCycle:
        cycles = None
        for layer, nodes in enumerate(nx.topological_generations(graph)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(graph, subset_key="layer")

    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, node_color="tab:blue", **options)

    if not cycles_only:
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            edgelist=graph.edges(),
            edge_color="tab:blue",
            alpha=0.5,
            width=3,
            label="S",
            arrowstyle="-|>",
        )

    if label_name is not None:
        nx.draw_networkx_labels(
            graph, pos, labels=labels, font_size=12, font_color="whitesmoke"
        )
    else:
        nx.draw_networkx_labels(graph, pos, font_size=12, font_color="whitesmoke")

    if cycles:
        nx.draw_networkx_edges(graph, pos, edgelist=cycles, edge_color="r", width=2)

    # plt.legend(fontsize = 'medium')
    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    return plt
