import json
import logging
from pathlib import Path
import os
from typing import List

import pandas as pd

from full_temporal_relation.data.postprocessing import prepare_df_from_response, majority_vote_decision
from full_temporal_relation.data.preprocessing import Doc, load_data
from full_temporal_relation.metrics import summary_results
from full_temporal_relation.models.HuggingFaceClient import HuggingfaceClient
from full_temporal_relation.models.LLModel import LLModel
# from full_temporal_relation.models.TogetherAIClient import TogetherAIClient
# from full_temporal_relation.models.gemini import Gemini
# from full_temporal_relation.models.llama3 import GroqModel
from full_temporal_relation.data.postprocessing import prepare_df_from_response, majority_vote_decision
from full_temporal_relation.visualization.graph import draw_directed_graph
from full_temporal_relation.data.preprocessing import load_data, Doc
from full_temporal_relation.graph import Graph, create_simple_graph
from prompts.Prompt import Prompt

DATA_PATH = Path('./data')
MATRES_DATA_PATH = DATA_PATH / 'MATRES'
PLATINUM_RAW = MATRES_DATA_PATH / 'raw' / 'TBAQ-cleaned' / 'te3-platinum'
gold_data_path = MATRES_DATA_PATH / 'platinum.txt'
TRC_RAW_PATH = DATA_PATH / 'TRC'
TRC_RESULTS_PATH = TRC_RAW_PATH / 'results'

def generate_suffix_name(model_name: str, method: str, suffix_path: str):
    suffixes = [model_name.split('/')[1] if '/' in model_name else model_name, method]
    if suffix_path:
        suffixes.append(suffix_path)
    return '-'.join(suffixes)

def main(model_name: str, method: str, model: LLModel,  prompt_params: List[str],
         raw_text_name: str, data_name: str, prompt: Prompt, suffix_path: str = '', mode: str = 'multi'):
    suffix_name = generate_suffix_name(model_name, method, suffix_path)

    llm_response_path = TRC_RAW_PATH / 'llm_response' / method / f'{mode}-{data_name}-{suffix_name}.jsonl'
    parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'{mode}-{data_name}-results-{suffix_name}.csv'
    results_path = TRC_RAW_PATH / 'results' / method / f'{mode}-{data_name}results-{suffix_name}.csv'

    # Generate model and response
    model.generate_responses(text_path=TRC_RAW_PATH / 'raw_text' / raw_text_name,
                            #  prompt_path=TRC_RAW_PATH / 'prompts' / method / prompt_filename,
                             results_path=llm_response_path,
                             prompt_params=prompt_params, 
                             prompt_template=prompt)

    all_parsed_response_df = pd.DataFrame(columns=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2',
                                                   'relation', 'unique_id', 'model_name',
                                                   'p_label', 'mode']).astype({'unique_id': str})

    # if mode == 'pair':
    #     records = json.load((TRC_RAW_PATH / 'raw_text' / raw_text_name).open('r'))
    #     responses = pd.read_json(llm_response_path, lines=True).to_dict(orient='records')
    #     for (record, response) in zip(records, responses):
    #         response['response'] = record['relations'].replace('[RELATION]', response['response'].split('\n\n')[0])
    #     results_df = pd.DataFrame.from_dict(responses)
    # else:
    #     results_df = pd.read_json(llm_response_path, lines=True)
    results_df = pd.read_json(llm_response_path, lines=True)

    for doc_id, group in results_df.groupby('doc_id'):
        doc = Doc(PLATINUM_RAW / f'{doc_id}.tml')
        for idx, (_, row) in enumerate(group.iterrows()):
            response = row.response

            parsed_response_df = prepare_df_from_response(response, doc, mode)
            parsed_response_df['model_name'] = model_name
            parsed_response_df['iter'] = idx
            all_parsed_response_df = pd.concat([all_parsed_response_df, parsed_response_df], ignore_index=True)

    all_parsed_response_df.to_csv(parsed_response_path, index=False)

    # parse model responses
    min_votes: int = 3
    majority_vote_decision(parsed_response_path, results_path, min_votes)
    return results_path

def get_summary_results(model_name: str, method: str, labeled_path: Path, results_path: Path, suffix_path: str = '') -> pd.DataFrame:
    result_file_suffix = generate_suffix_name(model_name, method, suffix_path)

    gold_df = load_data(labeled_path)

    # try:
    df = summary_results(results_path,
                         gold_df,
                         model_name)
    df['method'] = method
    df['suffix_path'] = suffix_path
    df['model_name'] = model_name
    return df


if __name__ == '__main__':
    gpu_device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    BASE_NAMES = ['AQUAINT', 'TimeBank', 'te3-platinum']
    BASE_DF_PATHS = ['aquaint.txt', 'timebank.txt', 'platinum.txt']

    name = 'te3-platinum'
    labeled_df_name = 'platinum.txt'

    mode = 'pair'
    # mode = 'multi'

    # method = 'zero-shot'
    method = 'few-shot'

    # prompt_filename = 'graph-generation-v1.txt'
    # prompt_params = ['text']
    # suffix_path = ''
    # prompt = Prompt()
    # prompt = Prompt()

    # prompt_filename = 'graph-generation-v2.txt'
    prompt_params = ['text', 'relations']
    suffix_path = 'completion'
    prompt = Prompt(use_few_shot=True, use_completion=True)

    # prompt_filename = 'graph-generation-v3.txt'
    # prompt_params = ['text', 'relations']
    # suffix_path = 'completion-explanation'
    # prompt = Prompt(use_few_shot=True, provide_justification=True, use_completion=True)

    # raw_text_name = 'platinum_text_prepared.json'
    # raw_text_name = 'platinum_text_w_relations_prepared.json'
    raw_text_name = f'{mode}_{name.lower()}_text_w_relations_prepared.json'

    # model_name = 'llama-3.1-70b-versatile'
    # model = GroqModel(model_name)

    # model_name = 'gemini-1.5-pro'
    # model_name = 'gemini-1.5-flash'
    # model = Gemini(model_name, n_trails=5)

    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # model = TogetherAIClient(model_name)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = HuggingfaceClient(model_name=model_name, device=gpu_device)

    """
    train - timebank.txt
    valid - aquaint.txt
    test - platinum.txt
    """

    results_path = main(model_name, method, model, prompt_params,
         raw_text_name=raw_text_name, suffix_path=suffix_path, data_name=name, mode=mode, prompt=prompt)

    results_df = get_summary_results(model_name, method,
                                  labeled_path=MATRES_DATA_PATH / labeled_df_name,
                                  results_path=results_path,
                                  suffix_path=suffix_path)

    results_metrics_path = TRC_RAW_PATH / 'final_metrics' / method / results_path.name
    results_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_metrics_path, index=False)