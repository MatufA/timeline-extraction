import abc
import json
from typing import List

import pandas as pd
from time import sleep
from pathlib import Path
from tqdm.auto import tqdm, trange


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

    def generate_responses(self, text_path: Path, prompt_path: Path,
                           results_path: Path, prompt_params: List[str] = None):
        if prompt_params is None:
            prompt_params = ['text']

        records = json.load(text_path.open('r'))
        prompt_template = prompt_path.open('r').read()

        # if results_path.exists():
        #     records_partial_df: pd.DataFrame = pd.read_json(results_path, lines=True)
        #
        #     if not records_partial_df.empty:
        #         # doc_iter = records_partial_df.groupby('doc_id')['n_tokens'].count().to_dict()
        #         doc_iter = records_partial_df.doc_id.unique()
        #         records = (record for record in records if record['doc_id'] not in doc_iter)

        with results_path.open('a') as file:
            for record in tqdm(records, desc='Text evaluation', position=0, leave=True):
                for trail in trange(self.n_trails, desc=f'Processing record: {record["doc_id"]}',
                                    position=1, leave=False):
                    prompt = prompt_template.format(**{p: record[p] for p in prompt_params})
                    response = self.generate_response(prompt)

                    res = self.prepare_response(response)
                    res['text'] = record['text']
                    res['doc_id'] = record['doc_id']
                    res['trail'] = trail
                    json_line = json.dumps(res)
                    file.write(json_line + '\n')

                    sleep(30) if self.each_trail else None

                sleep(10) if self.each_doc else None
