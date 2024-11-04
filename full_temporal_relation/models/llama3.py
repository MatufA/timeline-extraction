import json
import os
from pathlib import Path
from time import sleep

import pandas as pd
from groq import Groq
from tqdm import tqdm, trange

from full_temporal_relation.data.postprocessing import prepare_df_from_response
from full_temporal_relation.data.preprocessing import Doc
from full_temporal_relation.models.LLModel import LLModel


class GroqModel(LLModel):
    def __init__(self, model_name: str, n_trails: int = 5):
        model = self._initial_groq()
        self.model_name: str = model_name
        super().__init__(model, each_trail=False, each_doc=True, n_trails=n_trails)

    @staticmethod
    def _initial_groq() -> Groq:
        return Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def generate_response(self, prompt):
        return self.model.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model_name,
                )

    def prepare_response(self, response):
        return {
            'response': response.choices[0].message.content
        }




def generate_responses(model: Groq, platinum_text_prepared_path: Path, prompt_path: Path,
                       results_path: Path, model_name:str):
    records = json.load(platinum_text_prepared_path.open('r'))
    prompt_template = prompt_path.open('r').read()

    if results_path.exists():
        records_partial_df = pd.read_json(results_path, lines=True)
        if not records_partial_df.empty:
            doc_iter = records_partial_df.groupby('doc_id')['n_tokens'].count().to_dict()
            records = (record for record in records if record['doc_id'] not in doc_iter)

    with results_path.open('a') as file:
        for record in tqdm(records, desc='Platinum Text evaluation', position=0):
            for trail in trange(2, desc=f'Processing record: {record["doc_id"]}', position=1, leave=False):
                prompt = prompt_template.format(text=record['text'])
                response = model.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_name,
                )

                record['response'] = response.choices[0].message.content
                record['trail'] = trail
                json_line = json.dumps(record)
                file.write(json_line + '\n')
            sleep(10)


if __name__ == '__main__':
    model_name = 'llama-3.1-70b-versatile'
    method = 'zero-shot'  # 'few-shot'

    MATRES_DATA_PATH = Path('../../data')
    TRC_RAW_PATH = MATRES_DATA_PATH / 'TRC'
    results_path = TRC_RAW_PATH / 'llm_response' / method / f'platinum-test-{model_name}.jsonl'
    PLATINUM_RAW = MATRES_DATA_PATH / 'MATRES' / 'raw' / 'TBAQ-cleaned' / 'te3-platinum'
    parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'platinum-test-results-{model_name}.csv'

    # client = Groq(
    #     api_key=os.environ.get("GROQ_API_KEY"),
    # )
    #
    # generate_responses(client,
    #                    platinum_text_prepared_path=TRC_RAW_PATH / 'raw_text' / 'platinum_text_prepared.json',
    #                    prompt_path=TRC_RAW_PATH / 'prompts' / method / 'graph-generation-v1.txt',
    #                    results_path=results_path,
    #                    model_name=model_name)

    all_parsed_response_df = pd.DataFrame(columns=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2',
                                                   'relation', 'unique_id', 'model_name']).astype({'unique_id': str})
    results_df = pd.read_json(results_path, lines=True)
    for doc_id, group in results_df.groupby('doc_id'):
        doc = Doc(PLATINUM_RAW / f'{doc_id}.tml')
        for _, row in group.iterrows():
            text = row.text
            response = row.response

            parsed_response_df = prepare_df_from_response(response, doc)
            parsed_response_df['model_name'] = model_name
            parsed_response_df['iter'] = row.trail
            all_parsed_response_df = pd.concat([all_parsed_response_df, parsed_response_df], ignore_index=True)

    all_parsed_response_df.to_csv(parsed_response_path, index=False)
