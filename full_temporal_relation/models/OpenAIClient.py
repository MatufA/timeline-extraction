import json
import logging
from pathlib import Path
from typing import List
from full_temporal_relation.models.LLModel import LLModel, parse_dot_graph

from openai import OpenAI, api_key
from pydantic import BaseModel, Field
import os
import re
from tqdm.auto import tqdm

from prompts.Prompt import PairwisePrompt


class TemporalRelation(BaseModel):
    """ represantation of temporal event """
    event1: str = Field(..., description="the first relation, should be the lower #ID")
    event2: str = Field(..., description="the second relation, should be the higher #ID")
    relation: str = Field(..., description="the relation classification, one of before|after|equal|vague")

class TemporalRelations(BaseModel):
    """ list of temporal events """
    relations : List[TemporalRelation] = Field(..., description="a list of temporal relation, which represent the text as timeline graph")


class OpenAIClient(LLModel):
    def __init__(self, model_name: str = "gpt-4o-mini", use_formate: bool = False, use_dot_graph: bool = False , *args, **kwargs):
        self.model_name: str = model_name
        self.use_formate = use_formate
        self.use_dot_graph = use_dot_graph
        super().__init__(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), 
                         each_trail=False, 
                         each_doc=True,
                         *args, **kwargs)

    def generate_response(self, prompt):
        messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ] if isinstance(prompt, str) else prompt
        
        if self.use_formate:
            return self.model.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages
                , max_tokens=5000
                , temperature=0.7
                , n=5
                , response_format=TemporalRelations
                )
        
        return self.model.chat.completions.create(
            model=self.model_name,
            messages=messages
            , max_tokens=5000
            , temperature=0.7
            , n=5
        )

    def prepare_response(self, response):
        if self.use_formate:
            return [{
            'trail': idx,
            'response': [resp.__dict__ for resp in choice.message.parsed.relations] ,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        } for idx, choice in enumerate(response.choices)]

        if self.use_dot_graph:
            return [{
                'trail': idx,
                'response': parse_dot_graph(dot_graph=choice.message.content),
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            } for idx, choice in enumerate(response.choices)]
        
        return [{
                'trail': idx,
                'response': choice.message.content,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            } for idx, choice in enumerate(response.choices)]
    

def generate_batch_inference_file(mode: str, records_base_path: Path, use_few_shot: bool, use_vague: bool, 
                             is_full_text: bool, model_name: str = "gpt-4o-mini", n_trails: int = 5):
    context_ref = 'full_context' if is_full_text else 'minimal_context'
    records_path = records_base_path / f'{mode}_te3-platinum_{context_ref}_text_w_relations_prepared.json'
    output_batch_file = records_base_path.parent.parent / 'gpt_batch_{model_name}_te3-platinum_{context_ref}_format.jsonl'

    records = json.load(records_path.open('r'))
    prompt_template = PairwisePrompt(use_few_shot=use_few_shot, use_vague=use_vague)
    
    with output_batch_file.open('w') as obf:
        for record in tqdm(records, desc='Text evaluation', position=0, leave=True):
            prompt = prompt_template.generate_dict_prompt(text=record['text'])
            for trail in n_trails:
                line_format ={
                    "custom_id": '-'.join([trail, record['doc_id']] + record['relations'][0]), 
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": {
                        "model": model_name, 
                        "messages": prompt,
                        "tempertarure": 0.7
                        }
                    }
                json_line = json.dumps(line_format)
                obf.write(json_line + '\n')


def push_batch_job(batch_file: Path):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    batch_input_file = client.files.create(
        file=batch_file.open("rb"),
        purpose="batch"
        )
    print(f'batch_input_file={batch_input_file}')

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
            })

    print(f'batch job={batch}')


if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model = HuggingfaceClient(model_name=model_name)
