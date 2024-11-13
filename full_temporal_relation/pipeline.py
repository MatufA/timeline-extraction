import os
from typing import List

import numpy as np
import pandas as pd
from pathlib import Path

from full_temporal_relation.models.HuggingFaceClient import HuggingfaceClient
from full_temporal_relation.models.LLModel import LLModel
# from full_temporal_relation.models.TogetherAIClient import TogetherAIClient
# from full_temporal_relation.models.gemini import Gemini
# from full_temporal_relation.models.llama3 import GroqModel
from full_temporal_relation.data.postprocessing import prepare_df_from_response, majority_vote_decision
from full_temporal_relation.visualization.graph import draw_directed_graph
from full_temporal_relation.data.preprocessing import load_data, Doc
from full_temporal_relation.graph import Graph, create_simple_graph


def main(model_name: str, method: str, model: LLModel, prompt_filename: str,  prompt_params: List[str],
         raw_text_name: str, suffix_path: str = ''):
    DATA_PATH = Path('./data')

    MATRES_DATA_PATH = DATA_PATH / 'MATRES'
    PLATINUM_RAW = MATRES_DATA_PATH / 'raw' / 'TBAQ-cleaned' / 'te3-platinum'
    gold_data_path = MATRES_DATA_PATH / 'platinum.txt'

    suffixes = [model_name.split('/')[1] if '/' in model_name else model_name, method]
    if suffix_path:
        suffixes.append(suffix_path)
    suffix_name = '-'.join(suffixes)

    TRC_RAW_PATH = DATA_PATH / 'TRC'
    llm_response_path = TRC_RAW_PATH / 'llm_response' / method / f'platinum-test-{suffix_name}.jsonl'
    parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'platinum-test-results-{suffix_name}.csv'
    results_path = TRC_RAW_PATH / 'results' / method / f'platinum-results-{suffix_name}.csv'

    # Generate model and response
    model.generate_responses(text_path=TRC_RAW_PATH / 'raw_text' / raw_text_name,
                             prompt_path=TRC_RAW_PATH / 'prompts' / method / prompt_filename,
                             results_path=llm_response_path,
                             prompt_params=prompt_params)

    all_parsed_response_df = pd.DataFrame(columns=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2',
                                                   'relation', 'unique_id', 'model_name',
                                                   'p_label']).astype({'unique_id': str})
    results_df = pd.read_json(llm_response_path, lines=True)
    for doc_id, group in results_df.groupby('doc_id'):
        doc = Doc(PLATINUM_RAW / f'{doc_id}.tml')
        for idx, (_, row) in enumerate(group.iterrows()):
            response = row.response

            parsed_response_df = prepare_df_from_response(response, doc)
            parsed_response_df['model_name'] = model_name
            parsed_response_df['iter'] = idx
            all_parsed_response_df = pd.concat([all_parsed_response_df, parsed_response_df], ignore_index=True)

    all_parsed_response_df.to_csv(parsed_response_path, index=False)

    # parse model responses
    min_votes: int = 3
    majority_vote_decision(parsed_response_path, gold_data_path, results_path, min_votes)


if __name__ == '__main__':
    gpu_device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    method = 'zero-shot'
    # method = 'few-shot'

    prompt_filename = 'graph-generation-v1.txt'
    prompt_params = ['text']
    suffix_path = ''

    # prompt_filename = 'graph-generation-v2.txt'
    # prompt_params = ['text', 'relations']
    # suffix_path = 'completion'

    # prompt_filename = 'graph-generation-v3.txt'
    # prompt_params = ['text', 'relations']
    # suffix_path = 'completion-explanation'

    # raw_text_name = 'platinum_text_prepared.json'
    raw_text_name = 'platinum_text_w_relations_prepared.json'

    # model_name = 'llama-3.1-70b-versatile'
    # model = GroqModel(model_name)

    # model_name = 'gemini-1.5-pro'
    # model_name = 'gemini-1.5-flash'
    # model = Gemini(model_name, n_trails=5)

    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # model = TogetherAIClient(model_name)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = HuggingfaceClient(model_name=model_name, device=gpu_device)

    main(model_name, method, model, prompt_filename, prompt_params,
         raw_text_name=raw_text_name, suffix_path=suffix_path)
