import os
from typing import Union

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llama_model(path: Union[str, os.PathLike]):
    tokenizer = AutoTokenizer.from_pretrained(path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    return pipeline, tokenizer


if __name__ == '__main__':

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # pipeline, tokenizer = load_llama_model(model)
    #
    # sequences = pipeline(
    #     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    #     do_sample=True,
    #     top_k=1,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_length=200,
    # )
    #
    # for seq in sequences:
    #     print(f"Result: {seq['generated_text']}")

    # Load model directly

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokens = tokenizer.tokenize("hello how are you today?")
    print(len(tokens))
