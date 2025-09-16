import json
import tiktoken
from pathlib import Path
from typing import List

from timeline_extraction.prompts.Prompt import PairwisePrompt


def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        # print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        # print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_tokens(
    text_path: Path,
    prompt_params: List[str],
    model_name: str,
    is_few_shot: bool = False,
):
    prompt_template = PairwisePrompt(use_few_shot=is_few_shot, use_vague=False)

    records = json.load(text_path.open("r"))

    total_tokens = 0
    for record in records:
        messages = prompt_template.generate_dict_prompt(
            **{p: record[p] for p in prompt_params}
        )
        total_tokens += num_tokens_from_messages(messages, model=model_name)

    return total_tokens


if __name__ == "__main__":
    prompt_params = ["text"]
    DATA_PATH = Path("./data")
    TRC_RAW_PATH = DATA_PATH / "TRC"
    name = "te3-platinum"
    modes = ["comb", "pair"]

    for mode in modes:
        for is_few_shot in [True, False]:
            for model_name in [
                "gpt-3.5-turbo",
                "gpt-4-0613",
                "gpt-4",
                "gpt-4o",
                "gpt-4o-mini",
            ]:
                raw_text_name = f"{mode}_{name.lower()}_text_w_relations_prepared.json"
                text_path = TRC_RAW_PATH / "raw_text" / raw_text_name
                n_tokens = count_tokens(
                    text_path, prompt_params, model_name, is_few_shot
                )
                print(
                    f"mode={mode}, model name={model_name}, n_tokens={n_tokens}, is_few_shot={is_few_shot}"
                )
