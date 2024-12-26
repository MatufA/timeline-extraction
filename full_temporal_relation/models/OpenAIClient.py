import json
import logging
from typing import List
from full_temporal_relation.models.LLModel import LLModel, parse_dot_graph

from openai import OpenAI, api_key
from pydantic import BaseModel, Field
import os
import re


class TemporalRelation(BaseModel):
    """ represantation of temporal event """
    event1: str = Field(..., description="the first relation, should be the lower #ID")
    event2: str = Field(..., description="the second relation, should be the higher #ID")
    relation: str = Field(..., description="the relation classification, one of before|after|equal|vague")

class TemporalRelations(BaseModel):
    """ list of temporal events """
    relations : List[TemporalRelation] = Field(..., description="a list of temporal relation, which represent the text as timeline graph")


class OpenAIClient(LLModel):
    def __init__(self, model_name: str = "gpt-4o-mini", use_formate: bool = False, *args, **kwargs):
        self.model_name: str = model_name
        self.use_formate = use_formate
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

        return [{
            'trail': idx,
            'response': parse_dot_graph(dot_graph=choice.message.content),
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        } for idx, choice in enumerate(response.choices)]


if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model = HuggingfaceClient(model_name=model_name)
