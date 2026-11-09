"""Real local generation catches prompt leakage and token-budget regressions."""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from timeline_extraction.models.HuggingFaceClient import HuggingfaceClient


def test_local_generation_excludes_prompt_and_obeys_completion_budget(tmp_path):
    vocabulary = {"BEFORE": 0, "AFTER": 1, "user": 2, "assistant": 3, "Reply": 4, "unknown": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="unknown"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="unknown")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.save_pretrained(tmp_path)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocabulary), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, eos_token_id=None,
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(tmp_path)
    client = HuggingfaceClient(str(tmp_path), max_tokens=4, n_trails=1, device="cpu")
    result = client.prepare_response(client.generate_response("Reply BEFORE AFTER"))[0]
    assert result["response"] == "BEFORE BEFORE BEFORE BEFORE"
    assert result["prompt_tokens"] == 5
    assert result["completion_tokens"] == 4
    assert result["total_tokens"] == 9
