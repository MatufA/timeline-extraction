import json
import logging
import torch
from full_temporal_relation.models.LLModel import LLModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm, trange

from transformers import pipeline


class HuggingfaceClient(LLModel):
    def __init__(self, model_name: str, device: int, *args, **kwargs):
        self.model_name: str = model_name
        super().__init__(None, each_trail=False, each_doc=False, *args, **kwargs)
        self.pipe = pipeline("text-generation", 
                             model=model_name, 
                             torch_dtype=torch.float16, 
                             device_map='auto')

    def generate_response(self, prompt):
        messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ] if isinstance(prompt, str) else prompt
        responses = []
        for _ in trange(self.n_trails, desc=f'Processing record:', position=1, leave=False):
            response = self.pipe(
            messages
            , max_new_tokens=8000
            , pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            responses.append(response)
        return responses

    def prepare_response(self, response):
        responses = []
        for trail, res in enumerate(response):
            content = res[0]['generated_text'][-1]['content']
            # try:
            #     res = json.loads(content.replace('\n', ''))
            # except json.JSONDecodeError as e:
            #     try:
            #         res = json.loads(content.replace(',\n', '').replace('\n', ''))
            #     except json.JSONDecodeError as e:
            #         res = content

            responses.append({
            'content': content,
            'response': res, 
            'trail': trail
        })
        return responses


if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model = HuggingfaceClient(model_name=model_name)
