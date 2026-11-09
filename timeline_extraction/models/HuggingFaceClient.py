import os
from typing import Optional, Type

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from timeline_extraction.models.LLModel import LLModel, LabelParser, Parser
from timeline_extraction.models._responses import chat_messages


class HuggingfaceClient(LLModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        n_trails: int = 5,
        parser: Type[Parser] = LabelParser,
        *,
        device: Optional[str | int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(
            None,
            each_trail=False,
            each_doc=False,
            n_trails=n_trails,
            parser=parser,
        )
        download_options = {
            "token": api_key or os.environ.get("HF_TOKEN"),
            "cache_dir": cache_dir,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name, **download_options)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", **download_options
        )
        self.pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=device
        )
        self.model = model
        self.generation_config = GenerationConfig.from_dict(
            model.generation_config.to_dict(),
            max_new_tokens=max_tokens,
            max_length=None,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    def generate_response(self, prompt):
        messages = chat_messages(prompt)
        prompt_tokens = len(
            self.pipe.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
        )
        options = {
            "generation_config": self.generation_config,
            "return_tensors": True,
            "continue_final_message": False,
        }
        responses = []
        for _ in range(self.n_trails):
            response = self.pipe(messages, **options)
            response[0]["prompt_tokens"] = prompt_tokens
            responses.append(response)
        return responses

    def prepare_response(self, responses):
        results = []
        for trail, response in enumerate(responses):
            generated = response[0]
            prompt_tokens = generated["prompt_tokens"]
            token_ids = generated["generated_token_ids"]
            completion_ids = token_ids[prompt_tokens:]
            tokenizer = self.pipe.tokenizer
            if getattr(tokenizer, "response_template", None):
                text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                prefix = tokenizer.decode(token_ids[:prompt_tokens], skip_special_tokens=False)
                content = tokenizer.parse_response(text, prefix=prefix)["content"]
            else:
                content = tokenizer.decode(completion_ids, skip_special_tokens=True)
            if not content.strip():
                raise ValueError("HuggingFace returned no text content")
            results.append(
                {
                    "trail": trail,
                    "response": content,
                    "content": content,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(completion_ids),
                    "total_tokens": len(token_ids),
                }
            )
        return results
