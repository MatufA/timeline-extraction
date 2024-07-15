import pandas as pd
from pathlib import Path

from .data.postprocessing import prepare_df_from_response
from .visualization.graph import draw_directed_graph
from .data.preprocessing import load_data, Doc
from .graph import Graph, create_simple_graph

if __name__ == '__main__':
    MATRES_DATA_PATH = Path('../data') / 'MATRES'
    AQUAINT_RAW = MATRES_DATA_PATH / 'raw' / 'TBAQ-cleaned' / 'AQUAINT'

    aquaint_df = load_data(MATRES_DATA_PATH / 'aquaint.txt')
    APW19980807_0261 = Doc(AQUAINT_RAW / 'APW19980807.0261.tml')
    graph = Graph()

    APW19980807_0261_graph = graph.generate_directed_graph(df=aquaint_df[aquaint_df.docid == 'APW19980807.0261'])
    APW19980807_0261_graph_simple, _ = create_simple_graph(graph=APW19980807_0261_graph)
    draw_directed_graph(APW19980807_0261_graph_simple, title='expected timeline', label_name='eid')

    response = """"""
    llm_answer_df = prepare_df_from_response(response, APW19980807_0261)

    llm_answer_df_graph = graph.generate_directed_graph(df=llm_answer_df)
    llm_answer_df_graph_simple, _ = create_simple_graph(graph=llm_answer_df_graph)
    draw_directed_graph(llm_answer_df_graph_simple, title='predicted timeline', label_name='eid')
