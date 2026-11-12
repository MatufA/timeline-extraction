"""Consumer-facing response normalization regressions, without hosted API calls."""

from types import SimpleNamespace

import pytest
from google.genai import types
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from timeline_extraction.config import ConfigManager, ModelConfig
from timeline_extraction.models._reasoning import reasoning_options
from timeline_extraction.models.OpenAIClient import OpenAIClient
from timeline_extraction.models.TogetherAIClient import TogetherAIClient
from timeline_extraction.models.gemini import Gemini
from timeline_extraction.models.llama3 import GroqModel


@pytest.mark.parametrize("provider,model,effort", [("openai", "gpt-4o", "high"), ("openai", "gpt-6.1-sol", "none"), ("openai", "gpt-6-astra", "none"), ("openai", "gpt-6-astra", "minimal"), ("google", "gemini-3.8-flash", "minimal"), ("google", "gemini-3.8-flash", "none"), ("google", "gemini-decision-1", "high"), ("groq", "openai/gpt-oss-120b", "xhigh"), ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "high"), ("huggingface", "any", "high")])
def test_unsupported_reasoning_is_explicit(provider, model, effort):
    with pytest.raises(ValueError):
        ModelConfig(name=model, provider=provider, reasoning_effort=effort)


def test_default_reasoning_and_future_model_names_remain_unrestricted():
    assert reasoning_options("huggingface", "any", None) == {}
    assert reasoning_options("openai", "future-reasoning-model", "max") == {"reasoning": {"effort": "max"}}
    manager = ConfigManager()
    manager._parse_config({"models": {"openai": {"name": "gpt-6-luna", "reasoning_effort": "high"}}})
    assert manager.model_configs["openai"].reasoning_effort == "high"


def chat_completion(content="before", usage=None, finish_reason="stop", refusal=None):
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": refusal,
                    },
                }
            ],
            "usage": usage,
        }
    )


@pytest.mark.parametrize("adapter", [TogetherAIClient, GroqModel])
def test_chat_trials_have_independent_content_and_usage(adapter):
    model = adapter.__new__(adapter)
    first = chat_completion(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    second = chat_completion("after", {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})
    results = model.prepare_response([first, second])
    assert [(row["trail"], row["content"], row["total_tokens"]) for row in results] == [
        (0, "before", 12), (1, "after", 14)
    ]
    results[0]["response"] = "changed"
    assert results[1]["response"] == "after"


def test_missing_usage_is_unknown_not_fabricated():
    model = GroqModel.__new__(GroqModel)
    result = model.prepare_response([chat_completion()])[0]
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["total_tokens"] is None


@pytest.mark.parametrize(
    "response, error",
    [
        (chat_completion(None), "no text content"),
        (chat_completion("", refusal="Cannot comply"), "refused"),
        (chat_completion("bef", finish_reason="length"), "did not complete"),
    ],
)
def test_invalid_chat_outputs_do_not_become_classifications(response, error):
    model = TogetherAIClient.__new__(TogetherAIClient)
    with pytest.raises(ValueError, match=error):
        model.prepare_response([response])


def responses_completion(status="completed"):
    return Response.model_validate(
        {
            "id": "test",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": status,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "equal", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 8,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 12,
                "output_tokens_details": {"reasoning_tokens": 10},
                "total_tokens": 20,
            },
        }
    )


def test_responses_usage_includes_reasoning_tokens():
    model = OpenAIClient.__new__(OpenAIClient)
    model._chat_completions = False
    model.use_formate = False
    model.use_dot_graph = False
    result = model.prepare_response([responses_completion()])[0]
    assert result == {
        "trail": 0, "response": "equal", "content": "equal",
        "prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20,
    }


def test_incomplete_openai_output_is_not_silently_scored():
    model = OpenAIClient.__new__(OpenAIClient)
    model._chat_completions = False
    with pytest.raises(ValueError, match="incomplete"):
        model.prepare_response([responses_completion(status="incomplete")])


def test_gemini_preserves_conversation_roles_and_system_instructions():
    contents, instructions = Gemini._contents(
        [
            {"role": "system", "content": "Classify temporal relations."},
            {"role": "user", "content": "First example"},
            {"role": "assistant", "content": "before"},
            {"role": "user", "content": "Second example"},
        ]
    )
    assert instructions == "Classify temporal relations."
    assert [(item.role, item.parts[0].text) for item in contents] == [
        ("user", "First example"), ("model", "before"), ("user", "Second example")
    ]


def test_gemini_excludes_thought_text_but_counts_thought_tokens():
    model = Gemini.__new__(Gemini)
    raw = types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Maybe after?", "thought": True},
                            {"text": "before"},
                        ],
                    },
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 5, "candidates_token_count": 2,
                "thoughts_token_count": 7, "total_token_count": 14,
            },
        }
    )
    result = model.prepare_response([raw])[0]
    assert result["content"] == result["response"] == "before"
    assert result["completion_tokens"] == 9
    assert result["total_tokens"] == 14


def test_gemini_blocked_response_is_not_a_classification():
    model = Gemini.__new__(Gemini)
    raw = types.GenerateContentResponse.model_validate(
        {"prompt_feedback": {"block_reason": "SAFETY"}}
    )
    with pytest.raises(ValueError, match="no single text candidate"):
        model.prepare_response([raw])


def test_local_response_excludes_prompt_and_counts_generated_token_ids():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from timeline_extraction.models.HuggingFaceClient import HuggingfaceClient

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"prompt": 0, "before": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]")),
        eos_token="[EOS]", unk_token="[UNK]",
    )
    model = HuggingfaceClient.__new__(HuggingfaceClient)
    model.pipe = SimpleNamespace(tokenizer=tokenizer)
    result = model.prepare_response(
        [[{"prompt_tokens": 1, "generated_token_ids": [0, 1, 2]}]]
    )[0]
    assert result == {
        "trail": 0, "content": "before", "response": "before",
        "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
    }


def test_gemini_decision_builds_direct_subject_state_and_typed_choice_question():
    from timeline_extraction.prompts.Prompt import PairwisePrompt

    prompt = PairwisePrompt(use_vague=True, use_few_shot=False).generate_dict_prompt(
        text="Heavy snow is [EVENT2]causing[/EVENT2] disruption, with rainfall [EVENT5]bringing[/EVENT5] flooding.",
        relations=[["ei2", "ei5"]],
    )
    input_text, questions = Gemini._decision_request(prompt)
    assert input_text == (
        "Heavy snow is [EVENT2]causing[/EVENT2] disruption, with rainfall [EVENT5]bringing[/EVENT5] flooding."
    )
    assert "Task Overview" not in input_text
    assert "in one word" not in input_text
    question = questions["temporal_relation"]
    assert question["type"] == "choice"
    assert "[EVENT2]causing[/EVENT2]" in question["instructions"]
    assert "[EVENT5]bringing[/EVENT5]" in question["instructions"]
    assert set(question["criteria"]) == {"before", "after", "equal", "vague"}

    no_vague_prompt = PairwisePrompt(use_vague=False, use_few_shot=True).generate_dict_prompt(
        text="He [EVENT9]arrived[/EVENT9] after she [EVENT3]left[/EVENT3].",
        relations=[["ei9", "ei3"]],
    )
    no_vague_input, no_vague_questions = Gemini._decision_request(no_vague_prompt)
    assert no_vague_input.startswith("Examples:\n")
    assert no_vague_input.endswith("He [EVENT9]arrived[/EVENT9] after she [EVENT3]left[/EVENT3].")
    assert "[EVENT9]arrived[/EVENT9]" in no_vague_questions["temporal_relation"]["instructions"]
    assert "[EVENT3]left[/EVENT3]" in no_vague_questions["temporal_relation"]["instructions"]
    assert set(no_vague_questions["temporal_relation"]["criteria"]) == {"before", "after", "equal"}


def test_gemini_decision_normalizes_choice_probabilities_and_zero_output_tokens():
    import json
    from google.genai._gaos.types import decisions
    from timeline_extraction.prompts.Prompt import PairwisePrompt

    calls = []
    raw = decisions.CreateDecisionResponse.model_validate(
        {
            "answers": {
                "temporal_relation": {
                    "type": "choice",
                    "choice": "after",
                    "confidence": 0.91,
                    "probabilities": {
                        "after": 0.91,
                        "before": 0.04,
                        "equal": 0.03,
                        "vague": 0.02,
                    },
                }
            },
            "usage": {
                "total_input_tokens": 120,
                "total_output_tokens": 0,
                "total_tokens": 120,
            },
        }
    )
    model = Gemini.__new__(Gemini)
    model.model_name = "models/gemini-decision-1"
    model.is_decision_model = True
    model.n_trails = 1
    model.model = SimpleNamespace(
        decisions=SimpleNamespace(
            create=lambda **kwargs: (calls.append(kwargs) or raw)
        )
    )
    prompt = PairwisePrompt(use_vague=True).generate_dict_prompt(
        text="Heavy snow is [EVENT2]causing[/EVENT2] disruption, with rainfall [EVENT5]bringing[/EVENT5] flooding.",
        relations=[["ei2", "ei5"]],
    )
    responses = model.generate_response(prompt)
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-decision-1"
    assert calls[0]["api_version"] == "v1beta"
    result = model.prepare_response(responses)[0]
    assert result["response"] == "after"
    assert json.loads(result["content"]) == {
        "choice": "after",
        "confidence": 0.91,
        "probabilities": {
            "after": 0.91,
            "before": 0.04,
            "equal": 0.03,
            "vague": 0.02,
        },
    }
    assert result["prompt_tokens"] == 120
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 120


def test_gemini_decision_rejects_blocked_missing_or_multi_mode_usage():
    from google.genai._gaos.types import decisions
    from timeline_extraction.models.LLModel import JsonParser

    with pytest.raises(ValueError, match="only supports pairwise classification"):
        Gemini("gemini-decision-1", api_key="test", parser=JsonParser)

    model = Gemini.__new__(Gemini)
    blocked = decisions.CreateDecisionResponse.model_validate(
        {"prompt_feedback": {"block_reason": "SAFETY"}}
    )
    with pytest.raises(ValueError, match="decision request blocked"):
        model.prepare_response([blocked])

    empty = decisions.CreateDecisionResponse.model_validate({"answers": {}})
    with pytest.raises(ValueError, match="missing 'temporal_relation' answer"):
        model.prepare_response([empty])
