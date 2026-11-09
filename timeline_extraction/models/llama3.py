import os
from typing import Optional, Type

from groq import Groq

from timeline_extraction.models.LLModel import LLModel, LabelParser, Parser
from timeline_extraction.models._responses import chat_messages, chat_response
from timeline_extraction.models._reasoning import reasoning_options


class GroqModel(LLModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        n_trails: int = 5,
        parser: Type[Parser] = LabelParser,
        *,
        reasoning_effort: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_options = reasoning_options("groq", model_name, reasoning_effort)
        super().__init__(
            Groq(api_key=api_key or os.environ.get("GROQ_API_KEY")),
            each_trail=False,
            each_doc=True,
            n_trails=n_trails,
            parser=parser,
        )

    def generate_response(self, prompt):
        messages = chat_messages(prompt)
        return [
            self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                **getattr(self, "reasoning_options", {}),
            )
            for _ in range(self.n_trails)
        ]

    def prepare_response(self, responses):
        return [chat_response(response, trail) for trail, response in enumerate(responses)]
