import json
from pathlib import Path
from time import sleep
from typing import List

from typing import Optional
from timeline_extraction.models.LLModel import LLModel, parse_dot_graph

from openai import OpenAI
from pydantic import BaseModel, Field
import os
from tqdm.auto import tqdm

from timeline_extraction.prompts.Prompt import PairwisePrompt


class TemporalRelation(BaseModel):
    """represantation of temporal event"""

    event1: str = Field(..., description="the first relation, should be the lower #ID")
    event2: str = Field(
        ..., description="the second relation, should be the higher #ID"
    )
    relation: str = Field(
        ..., description="the relation classification, one of before|after|equal|vague"
    )


class TemporalRelations(BaseModel):
    """list of temporal events"""

    relations: List[TemporalRelation] = Field(
        ...,
        description="a list of temporal relation, which represent the text as timeline graph",
    )


class TemporalRelationsToDrop(BaseModel):
    """list of unique id of temporal relation to drop"""

    unique_ids: List[int] = Field(
        ..., description="a list of unique ids of temporal relation"
    )


class OpenAIClient(LLModel):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        use_formate: bool = False,
        use_dot_graph: bool = False,
        response_format: Optional[BaseModel] = None,
        *args,
        **kwargs,
    ):
        self.model_name: str = model_name
        self.use_formate = use_formate
        self.use_dot_graph = use_dot_graph
        self.response_format = response_format
        super().__init__(
            OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
            each_trail=False,
            each_doc=True,
            *args,
            **kwargs,
        )

    def generate_response(self, prompt):
        messages = (
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            if isinstance(prompt, str)
            else prompt
        )

        if self.response_format is not None:
            return self.model.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                max_tokens=5000,
                temperature=0.7,
                n=5,
                response_format=self.response_format,
            )

        return self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=5000,
            temperature=0.7,
            n=5,
        )

    def prepare_response(self, response):
        if self.use_formate:
            return [
                {
                    "trail": idx,
                    "response": [
                        resp.__dict__ for resp in choice.message.parsed.relations
                    ],
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                for idx, choice in enumerate(response.choices)
            ]

        if self.use_dot_graph:
            return [
                {
                    "trail": idx,
                    "response": parse_dot_graph(dot_graph=choice.message.content),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                for idx, choice in enumerate(response.choices)
            ]

        return [
            {
                "trail": idx,
                "response": choice.message.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            for idx, choice in enumerate(response.choices)
        ]


def generate_batch_inference_file(
    mode: str,
    records_base_path: Path,
    output_dir_path: Path,
    use_few_shot: bool,
    use_vague: bool,
    is_full_text: bool,
    model_name: str = "gpt-4o-mini",
    n_trails: int = 5,
):
    context_ref = "full_context" if is_full_text else "minimal_context"
    vague_ref = "w_vague" if use_vague else "wo_vague"
    records_path = (
        records_base_path
        / f"{mode}_te3-platinum_{context_ref}_text_w_relations_prepared.json"
    )
    output_batch_file = (
        output_dir_path
        / f"{mode}_{vague_ref}_gpt_batch_{model_name}_te3-platinum_{context_ref}_format.jsonl"
    )

    records = json.load(records_path.open("r"))
    prompt_template = PairwisePrompt(use_few_shot=use_few_shot, use_vague=use_vague)
    counter = 0

    with output_batch_file.open("w") as obf:
        unique_ids = set()
        for record in tqdm(records, desc="Text evaluation", position=0, leave=True):
            prompt = prompt_template.generate_dict_prompt(text=record["text"])
            for trail in range(n_trails):
                unique_id = "-".join(
                    [str(trail), record["doc_id"]] + record["relations"][0]
                )

                if unique_id in unique_ids:
                    print(f"duplicate {unique_id}")
                    continue

                unique_ids.add(unique_id)

                line_format = {
                    "custom_id": unique_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": prompt,
                        "tempertarure": 0.7,
                        "max_tokens": 5000,
                        "response_format": {"type": "json_object"},
                    },
                }
                json_line = json.dumps(line_format)
                obf.write(json_line + "\n")
                counter += 1


def push_batch_job(batch_file: Path):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    batch_input_file = client.files.create(file=batch_file.open("rb"), purpose="batch")
    print(f"batch_input_file={batch_input_file}")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )

    print(f"batch job={batch}")
    sleep(60)

    batch_job = client.batches.retrieve(batch_id=batch.id)

    status = batch_job.status
    print(f"status={status}")

    method = batch_file.parent.name
    if batch_job.output_file_id:
        file_response = client.files.content(batch_job.output_file_id)
        batch_file_name = batch_file.name.replace(".jsonl", "_results.jsonl")
    else:
        file_response = client.files.content(batch_job.error_file_id)
        batch_file_name = batch_file.name.replace(".jsonl", "_errors.jsonl")
    batch_file_out = (
        batch_file.parent.parent.parent / "output" / method / batch_file_name
    )

    with batch_file_out.open("w") as f:
        f.write(file_response.text)


if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    mode = "pair"  # all paires within minimal context
    mode = "comb"  # all posible paires in 2 line a part context
    # model = HuggingfaceClient(model_name=model_name)
    records_base_path = Path("/home/adiel/full-temporal-relation/data/TRC/raw_text")
    output_dir_path = Path("/home/adiel/full-temporal-relation/data/batch/input")
    output_res_dir_path = Path("/home/adiel/full-temporal-relation/data/batch/output")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_res_dir_path.mkdir(parents=True, exist_ok=True)

    model_name: str = "gpt-4o"

    for method in ["zero-shot", "few-shot"]:
        for use_vague in [True, False]:
            for is_full_text in [True, False]:
                output_method_path = output_dir_path / method
                output_method_path.mkdir(parents=True, exist_ok=True)
                # generate_batch_inference_file(mode=mode,
                #                             records_base_path=records_base_path,
                #                             output_dir_path=output_method_path,
                #                             use_few_shot=(method=='few-shot'),
                #                             use_vague=use_vague,
                #                             is_full_text=is_full_text,
                #                             model_name=model_name,
                #                             n_trails = 5)
                context_ref = "full_context" if is_full_text else "minimal_context"
                vague_ref = "w_vague" if use_vague else "wo_vague"
                push_batch_job(
                    output_method_path
                    / f"{mode}_{vague_ref}_gpt_batch_{model_name}_te3-platinum_{context_ref}_format.jsonl"
                )
