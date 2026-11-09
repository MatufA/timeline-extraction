import os
from typing import Optional, Type

from google import genai
from google.genai import types

from timeline_extraction.models.LLModel import LLModel, LabelParser, Parser
from timeline_extraction.models._responses import chat_messages
from timeline_extraction.models._reasoning import reasoning_options


class Gemini(LLModel):
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
        self.reasoning_options = reasoning_options("google", model_name, reasoning_effort)
        super().__init__(
            genai.Client(
                api_key=api_key
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            ),
            each_trail=False,
            each_doc=True,
            n_trails=n_trails,
            parser=parser,
        )

    @staticmethod
    def _contents(prompt):
        contents = []
        instructions = []
        for message in chat_messages(prompt):
            role = message["role"]
            text = message["content"]
            if not isinstance(text, str):
                raise ValueError("Gemini text prompts require string message content")
            if role in {"system", "developer"}:
                instructions.append(text)
            elif role in {"user", "assistant"}:
                contents.append(
                    types.Content(
                        role="model" if role == "assistant" else "user",
                        parts=[types.Part.from_text(text=text)],
                    )
                )
            else:
                raise ValueError(f"Unsupported Gemini message role: {role}")
        return contents, "\n\n".join(instructions) or None

    def generate_response(self, prompt):
        contents, instructions = self._contents(prompt)
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            system_instruction=instructions,
            thinking_config=types.ThinkingConfig(**self.reasoning_options) if getattr(self, "reasoning_options", {}) else None,
        )
        return [
            self.model.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            for _ in range(self.n_trails)
        ]

    def prepare_response(self, responses):
        results = []
        for trail, response in enumerate(responses):
            candidates = response.candidates or []
            if len(candidates) != 1:
                raise ValueError(
                    f"Gemini returned no single text candidate: {response.prompt_feedback}"
                )
            candidate = candidates[0]
            finish_reason = getattr(candidate.finish_reason, "value", candidate.finish_reason)
            if finish_reason not in {None, "STOP"}:
                raise ValueError(f"Gemini did not complete the response: {finish_reason}")
            parts = candidate.content.parts if candidate.content else []
            content = "".join(
                part.text
                for part in (parts or [])
                if part.text and not part.thought
            )
            if not content.strip():
                raise ValueError("Gemini returned no text content")
            usage = response.usage_metadata
            completion_tokens = getattr(usage, "candidates_token_count", None)
            if completion_tokens is not None:
                completion_tokens += getattr(usage, "thoughts_token_count", None) or 0
            results.append(
                {
                    "trail": trail,
                    "response": content,
                    "content": content,
                    "prompt_tokens": getattr(usage, "prompt_token_count", None),
                    "completion_tokens": completion_tokens,
                    "total_tokens": getattr(usage, "total_token_count", None),
                }
            )
        return results
