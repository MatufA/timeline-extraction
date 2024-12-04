import re
import abc
import json
from typing import List, Type

import pandas as pd
from time import sleep
from pathlib import Path
from tqdm.auto import tqdm, trange

from full_temporal_relation.prompts.Prompt import Prompt, MultiEvents, PairwisePrompt
from full_temporal_relation.data.preprocessing import replace_eid


class LLModel(abc.ABC):
    def __init__(self, model, each_trail: bool = False, each_doc: bool = True, n_trails: int = 5):
        self.model = model
        self.each_trail = each_trail
        self.each_doc = each_doc
        self.n_trails = n_trails

    @abc.abstractmethod
    def generate_response(self, prompt):
        pass

    @abc.abstractmethod
    def prepare_response(self, response):
        pass

    def generate_responses(self, text_path: Path, results_path: Path, prompt_template: Prompt, 
                           prompt_params: List[str] = None):
        if prompt_params is None:
            prompt_params = ['text']

        records = json.load(text_path.open('r'))

        new_records = []
        if 'relations' in prompt_params:
            prompt_params = ['text']
            # for record in records:
            #     rels = []
            #     for rel in record['relations'].split('\n'):
            #         e1, e2 = rel.split(' [RELATION] ')
            #         e1, e2 = e1.strip(), e2.strip()
            #         # rels.append(json.dumps({"event1": e1, "event2": e2}))
            #         rels.append(f'{e1} {e2}')
            #         new_records.append({
            #             'text': replace_eid(record['text'], exclude_ids=[e1, e2]),
            #             'doc_id': record["doc_id"], 
            #             'e1': e1,
            #             'e2': e2
            #             })


                # record['relations'] = '\n'.join(rels)

        # prompt_template = prompt_path.open('r').read()

        # if results_path.exists():
        #     records_partial_df: pd.DataFrame = pd.read_json(results_path, lines=True)
        #
        #     if not records_partial_df.empty:
        #         # doc_iter = records_partial_df.groupby('doc_id')['n_tokens'].count().to_dict()
        #         doc_iter = records_partial_df.doc_id.unique()
        #         records = (record for record in records if record['doc_id'] not in doc_iter)

        resluts = []
        with results_path.open('w') as file:
            for record in tqdm(records, desc='Text evaluation', position=0, leave=True):
                for trail in trange(self.n_trails, desc=f'Processing record: {record["doc_id"]}',
                                    position=1, leave=False):
                    # prompt = prompt_template.format(**{p: record[p] for p in prompt_params})
                    prompt = prompt_template.generate_dict_prompt(**{p: record[p] for p in prompt_params})
                    response = self.generate_response(prompt)

                    res = self.prepare_response(response)
                    res.update(record)
                    res['prompt'] = prompt
                    res['trail'] = trail
                    if isinstance(prompt_template, PairwisePrompt) and not isinstance(res['response'], dict) :
                        start_pattern = r'^(before|after|equal|vague)\b'
                        end_pattern = r'\b(before|after|equal|vague)(?:[.!?]|\s*$)'
                        if start_label_match := re.search(start_pattern, res['response'].lower()):
                            plabel = start_label_match.group(1)
                        elif end_label_match := re.search(end_pattern, res['response'].lower()):
                            plabel = end_label_match.group(1)
                        else:
                            plabel = res['response']

                        res['response'] = plabel
                    json_line = json.dumps(res)
                    file.write(json_line + '\n')
                    resluts.append(res)

                    sleep(30) if self.each_trail else None

                sleep(10) if self.each_doc else None
        return resluts
