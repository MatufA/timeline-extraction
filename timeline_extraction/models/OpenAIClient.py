import os
import re
from typing import List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel, Field

from timeline_extraction.models.LLModel import LLModel, LabelParser, Parser, parse_dot_graph
from timeline_extraction.models._responses import chat_messages, chat_response
from timeline_extraction.models._reasoning import reasoning_options


class TemporalRelation(BaseModel):
    """A classified temporal relation between two events."""

    event1: str = Field(..., description="the first event, with the lower #ID")
    event2: str = Field(..., description="the second event, with the higher #ID")
    relation: str = Field(..., description="one of before|after|equal|vague")


class TemporalRelations(BaseModel):
    """Temporal relations representing a timeline graph."""

    relations: List[TemporalRelation]


class TemporalRelationsToDrop(BaseModel):
    """Unique IDs of temporal relations to remove from a graph."""

    unique_ids: List[int]


class OpenAIClient(LLModel):
    def __init__(
        self,
        model_name: str = "gpt-6-luna",
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        n_trails: int = 5,
        parser: Type[Parser] = LabelParser,
        *,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        use_formate: bool = False,
        use_dot_graph: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_options = reasoning_options("openai", model_name, reasoning_effort)
        self.use_formate = use_formate
        self.use_dot_graph = use_dot_graph
        self.response_format = response_format or (TemporalRelations if use_formate else None)
        # Custom OpenAI-compatible endpoints generally implement Chat Completions.
        self._chat_completions = bool(base_url or os.environ.get("OPENAI_BASE_URL"))
        super().__init__(
            OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url),
            each_trail=False,
            each_doc=True,
            n_trails=n_trails,
            parser=parser,
        )

    def generate_response(self, prompt):
        reasoning = bool(re.match(r"^(?:o\d|gpt-(?:[5-9]|\d{2,}))", self.model_name))
        options = {"model": self.model_name}
        effort_options = getattr(self, "reasoning_options", {})
        if effort_options:
            if self._chat_completions:
                options["reasoning_effort"] = effort_options["reasoning"]["effort"]
            else:
                options.update(effort_options)
        if not reasoning or effort_options.get("reasoning", {}).get("effort") == "none":
            options["temperature"] = self.temperature
        if self._chat_completions:
            options["messages"] = chat_messages(prompt)
            options["max_completion_tokens" if reasoning else "max_tokens"] = self.max_tokens
            if self.response_format is not None:
                create = self.model.chat.completions.parse
                options["response_format"] = self.response_format
            else:
                create = self.model.chat.completions.create
        else:
            options["input"] = prompt
            options["max_output_tokens"] = self.max_tokens
            if self.response_format is not None:
                create = self.model.responses.parse
                options["text_format"] = self.response_format
            else:
                create = self.model.responses.create
        # Separate requests work for models without n and give per-trial usage.
        return [create(**options) for _ in range(self.n_trails)]

    def prepare_response(self, responses):
        results = []
        for trail, response in enumerate(responses):
            if self._chat_completions:
                result = chat_response(response, trail)
                parsed = getattr(response.choices[0].message, "parsed", None)
            else:
                if response.status != "completed":
                    details = getattr(response, "incomplete_details", None) or getattr(
                        response, "error", None
                    )
                    raise ValueError(f"OpenAI response {response.status}: {details}")
                for item in response.output:
                    for part in getattr(item, "content", []) or []:
                        if part.type == "refusal":
                            raise ValueError(f"OpenAI refused the request: {part.refusal}")
                content = response.output_text
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("OpenAI returned no text content")
                usage = response.usage
                result = {
                    "trail": trail,
                    "response": content,
                    "content": content,
                    "prompt_tokens": getattr(usage, "input_tokens", None),
                    "completion_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                parsed = getattr(response, "output_parsed", None)
            if self.use_formate:
                if parsed is None:
                    raise ValueError("OpenAI returned no parsed temporal relations")
                result["response"] = [relation.model_dump() for relation in parsed.relations]
            elif self.use_dot_graph:
                result["response"] = parse_dot_graph(dot_graph=result["content"])
            results.append(result)
        return results
