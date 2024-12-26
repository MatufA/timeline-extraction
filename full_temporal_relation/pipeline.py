import json
import logging
from pathlib import Path
import os
from typing import List

import pandas as pd
import torch
import gc
from dotenv import load_dotenv

from full_temporal_relation.data.postprocessing import majority_vote_decision_new
from full_temporal_relation.data.preprocessing import Doc, load_data
from full_temporal_relation.metrics import summary_results
from full_temporal_relation.models.HuggingFaceClient import HuggingfaceClient
from full_temporal_relation.models.OpenAIClient import OpenAIClient
from full_temporal_relation.models.LLModel import LLModel, LabelParser, JsonParser
# from full_temporal_relation.models.TogetherAIClient import TogetherAIClient
# from full_temporal_relation.models.gemini import Gemini
# from full_temporal_relation.models.llama3 import GroqModel
from full_temporal_relation.prompts.Prompt import Prompt, PairwisePrompt, MultiEvents

# load_dotenv()

DATA_PATH = Path('./data')
MATRES_DATA_PATH = DATA_PATH / 'MATRES'
PLATINUM_RAW = MATRES_DATA_PATH / 'raw' / 'TBAQ-cleaned' / 'te3-platinum'
gold_data_path = MATRES_DATA_PATH / 'platinum.txt'
TRC_RAW_PATH = DATA_PATH / 'TRC'
TRC_RESULTS_PATH = TRC_RAW_PATH / 'results'

def generate_suffix_name(model_name: str, method: str, suffix_path: str, use_vague: bool = False):
    suffixes = [model_name.split('/')[1] if '/' in model_name else model_name, method]
    if use_vague:
         suffixes.append('w_vague')
    if suffix_path:
        suffixes.append(suffix_path)
    return '-'.join(suffixes)

def main(model_name: str, method: str, model: LLModel,  prompt_params: List[str],
         raw_text_name: str, data_name: str, prompt: Prompt, suffix_path: str = '',
         mode: str = 'multi', overwrite: bool = False, use_vague: bool = False, 
         skip_model_eval: bool = False):
    suffix_name = generate_suffix_name(model_name, method, suffix_path, use_vague)

    llm_response_path = TRC_RAW_PATH / 'llm_response' / method / f'{mode}-{data_name}-{suffix_name}.jsonl'
    parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'{mode}-{data_name}-results-{suffix_name}.csv'
    results_path = TRC_RAW_PATH / 'results' / method / f'{mode}-{data_name}-results-{suffix_name}.csv'
    checkpoint_results_path = TRC_RAW_PATH / 'llm_raw_response' / method / f'{mode}-{data_name}-{suffix_name}.jsonl'
    checkpoint_results_path.parent.mkdir(parents=True, exist_ok=True) 

    # Generate model and response
    if not skip_model_eval:
        resluts = model.generate_responses(text_path=TRC_RAW_PATH / 'raw_text' / raw_text_name,
                                results_path=llm_response_path,
                                prompt_params=prompt_params, 
                                prompt_template=prompt, 
                                overwrite=overwrite, 
                                checkpoint_results=checkpoint_results_path)

    # results_df = pd.DataFrame({'doc_id': res['doc_id'], 'trail': res['trail'], 'response': res['response']} for res in resluts)

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


    data = []
    for idx, row in results_df.iterrows():
        if isinstance(prompt, PairwisePrompt):
            true_labels = [row.true_label.upper()] if 'true_label' in row else ['NO_TRUE_LABEL']
        else:
            true_labels = ['NO_TRUE_LABEL']

        if isinstance(row.response, str):
            p_label = row.response.upper()
        else:
            p_label = row.response['relation'].upper()

        for true_label, (e1, e2) in zip(true_labels, row.relations):
            data.append({
                'docid': row.doc_id,
                'eiid1': e1,
                'eiid2': e2,
                'relation': true_label,
                'unique_id': '-'.join(sorted([e1, e2])),
                'p_label': p_label,
                'mode': mode,
                'model_name': model_name,
                'iter': row.trail, 
                'prompt': row.prompt,
                'raw_response': row.content if 'content' in row else row.response
            })
    all_parsed_response_df = pd.DataFrame(data)
        

    # else:
    #     for doc_id, group in results_df.groupby('doc_id'):
    #         doc = Doc(PLATINUM_RAW / f'{doc_id}.tml')
    #         for idx, (_, row) in enumerate(group.iterrows()):
    #             response = row.response

    #             # parsed_response_df = prepare_df_from_response(response, doc, mode)
    #             parsed_response_df = prepare_df_from_json_response(response, doc, mode)
    #             parsed_response_df['model_name'] = model_name
    #             parsed_response_df['iter'] = idx
    #             all_parsed_response_df = pd.concat([all_parsed_response_df, parsed_response_df], ignore_index=True)

    all_parsed_response_df.to_csv(parsed_response_path, index=False)

    # parse model responses
    min_votes: int = 3
    selected_relations_df = majority_vote_decision_new(parsed_response_path, results_path, min_votes)

    return results_path

def get_summary_results(model_name: str, method: str, labeled_path: Path, results_path: Path, suffix_path: str = '') -> pd.DataFrame:
    result_file_suffix = generate_suffix_name(model_name, method, suffix_path)

    gold_df = load_data(labeled_path)
    # non_vague_gold_df = gold_df[gold_df['label'] != 'VAGUE']
    non_vague_gold_df = gold_df
    results_df = pd.read_csv(results_path)
    results_df = pd.merge(results_df, non_vague_gold_df[['docid', 'unique_id']], how='inner',
                         on=['docid', 'unique_id']).drop_duplicates(['docid', 'unique_id'])

    df = summary_results(results_df,
                         non_vague_gold_df,
                         model_name)
    df['method'] = method
    df['suffix_path'] = suffix_path
    df['model_name'] = model_name
    return df


if __name__ == '__main__':
    gpu_device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # for mistralai

    BASE_NAMES = ['AQUAINT', 'TimeBank', 'te3-platinum']
    BASE_DF_PATHS = ['aquaint.txt', 'timebank.txt', 'platinum.txt']

    name = 'te3-platinum'
    labeled_df_name = 'platinum.txt'

    # mode = 'pair'
    mode = 'multi'
    # mode = 'comb'

    methods = ['zero-shot', 'few-shot']
    # method = 'zero-shot'
    # method = 'few-shot'

    # prompt_filename = 'graph-generation-v1.txt'
    # prompt_params = ['text']
    # suffix_path = ''
    # prompt = Prompt()
    # prompt = Prompt()

    # prompt_filename = 'graph-generation-v2.txt'
    prompt_params = ['text', 'relations']
    suffix_path = 'completion'

    # prompt_filename = 'graph-generation-v3.txt'
    # prompt_params = ['text', 'relations']
    # suffix_path = 'completion-explanation'
    # prompt = MultiEvents(use_few_shot=True, provide_justification=False, use_completion=True, use_vague=False)

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
    
    # model_names = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"] 
    # model_names = ['mistralai/Mistral-7B-Instruct-v0.3']

    model_names = ['gpt-4o-mini'] 

    use_vague = True

    # parser_type = LabelParser
    parser_type = JsonParser
    overwrite=True

    for model_name in model_names:
        for method in methods:
            is_few_shot = (method == 'few-shot')

            if isinstance(parser_type, LabelParser):
                prompt = PairwisePrompt(use_few_shot=is_few_shot, use_vague=use_vague)
            else:
                prompt = MultiEvents(use_few_shot=is_few_shot, use_vague=use_vague, provide_justification=False)
            

            # model = HuggingfaceClient(model_name=model_name, device=gpu_device, parser=parser_type)
            model = OpenAIClient(model_name=model_name, use_formate=False, parser=parser_type)

            """
            train - timebank.txt
            valid - aquaint.txt
            test - platinum.txt
            """

            results_path = main(model_name, method, model, prompt_params,
                raw_text_name=raw_text_name, suffix_path=suffix_path, 
                data_name=name, mode=mode, prompt=prompt, overwrite=overwrite, skip_model_eval=False)

            results_df = get_summary_results(model_name, method,
                                        labeled_path=MATRES_DATA_PATH / labeled_df_name,
                                        results_path=results_path,
                                        suffix_path=suffix_path)

            results_metrics_path = TRC_RAW_PATH / 'final_metrics' / method / results_path.name
            results_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_metrics_path, index=False)
            
            del model
            torch.cuda.empty_cache()
            gc.collect()
            print(f'writing results to {results_metrics_path}')
            print('-' * 50)
