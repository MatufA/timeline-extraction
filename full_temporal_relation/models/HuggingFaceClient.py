import json
import logging
import torch
from full_temporal_relation.models.LLModel import LLModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline


class HuggingfaceClient(LLModel):
    def __init__(self, model_name: str, device: int, n_trails: int = 5):
        self.model_name: str = model_name
        super().__init__(None, each_trail=False, each_doc=False, n_trails=n_trails)
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
        return self.pipe(
            messages
            , max_new_tokens=500
            , pad_token_id=self.pipe.tokenizer.eos_token_id
        )

    def prepare_response(self, response):
        content = response[0]['generated_text'][-1]['content']
        try:
            response = json.loads(content.replace('\n', ''))
        except json.JSONDecodeError as e:
            try:
                response = json.loads(content.replace(',\n', '').replace('\n', ''))
            except json.JSONDecodeError as e:
                response = content

        return {
            'content': content,
            'response': response
        }


if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model = HuggingfaceClient(model_name=model_name)
