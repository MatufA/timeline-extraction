import torch
from full_temporal_relation.models.LLModel import LLModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")


class HuggingfaceClient(LLModel):
    def __init__(self, model_name: str, device: int, n_trails: int = 5):
        self.model_name: str = model_name
        super().__init__(None, each_trail=False, each_doc=False, n_trails=n_trails)
        self.pipe = pipeline("text-generation", 
                             model=model_name, 
                             torch_dtype=torch.float16, 
                             device=device)

    def generate_response(self, prompt):
        return self.pipe(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ], max_new_tokens=500
            , pad_token_id=self.pipe.tokenizer.eos_token_id
        )

    def prepare_response(self, response):
        return {
            'response': response[0]['generated_text'][1]['content']
        }


if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model = HuggingfaceClient(model_name=model_name)
