import json
import logging
from full_temporal_relation.models.LLModel import LLModel

from openai import OpenAI, api_key
import os


class OpenAIClient(LLModel):
    def __init__(self, model_name: str = "gpt-4o-mini", n_trails: int = 5):
        self.model_name: str = model_name
        super().__init__(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), 
                         each_trail=False, 
                         each_doc=True, 
                         n_trails=n_trails)

    def generate_response(self, prompt):
        messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ] if isinstance(prompt, str) else prompt
        
        return self.model.chat.completions.create(
            model=self.model_name,
            messages=messages
            , max_tokens=500
            , temperature=0.7
            , n=5
        )

    def prepare_response(self, response):
        return [{
            'response': choice.message.content,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        } for choice in response.choices]


if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model = HuggingfaceClient(model_name=model_name)
