import json
import os
from pathlib import Path
from time import sleep
from typing import List

import pandas as pd
from tqdm import tqdm, trange

import google.generativeai as genai

from timeline_extraction.data.postprocessing import prepare_df_from_response
from timeline_extraction.data.preprocessing import Doc
from timeline_extraction.models.LLModel import LLModel


class Gemini(LLModel):

    def __init__(
        self,
        model_name: str,
        max_output_tokens: int = 2000,
        stop_sequences: List[str] = None,
        n_trails: int = 5,
    ):
        if stop_sequences is None:
            stop_sequences = ["DONE!", "\n\n"]

        model = self._initial_google_genai(
            model_name,
            max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences,
        )
        each_trail = model.model_name.endswith("pro")
        each_doc = model.model_name.endswith("flash")
        super().__init__(model, each_trail, each_doc, n_trails)

    @staticmethod
    def _initial_google_genai(
        model_name: str, max_output_tokens: int, stop_sequences: list
    ) -> genai.GenerativeModel:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_output_tokens, stop_sequences=stop_sequences
            ),
        )

    def generate_response(self, prompt):
        return self.model.generate_content(prompt)

    def prepare_response(self, response):
        return {
            "response": response.text,
            "total_token_count": response.usage_metadata.total_token_count,
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "candidates_token_count": response.usage_metadata.candidates_token_count,
        }


def initial_google_genai(
    model_name: str, max_output_tokens: int, stop_sequences: list
) -> genai.GenerativeModel:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    return genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_output_tokens, stop_sequences=stop_sequences
        ),
    )


def generate_responses(
    model: genai.GenerativeModel,
    platinum_text_prepared_path: Path,
    prompt_path: Path,
    results_path: Path,
):
    records = json.load(platinum_text_prepared_path.open("r"))
    prompt_template = prompt_path.open("r").read()

    if results_path.exists():
        records_partial_df = pd.read_json(results_path, lines=True)
        doc_iter = records_partial_df.groupby("doc_id")["n_tokens"].count().to_dict()
        records = (record for record in records if record["doc_id"] not in doc_iter)

    with results_path.open("a") as file:
        for record in tqdm(records, desc="Platinum Text evaluation", position=0):
            for trail in trange(
                5,
                desc=f'Processing record: {record["doc_id"]}',
                position=1,
                leave=False,
            ):
                prompt = prompt_template.format(text=record["text"])
                response = model.generate_content(prompt)

                record["response"] = response.text
                record["total_token_count"] = response.usage_metadata.total_token_count
                record["prompt_token_count"] = (
                    response.usage_metadata.prompt_token_count
                )
                record["candidates_token_count"] = (
                    response.usage_metadata.candidates_token_count
                )
                record["trail"] = trail
                json_line = json.dumps(record)
                file.write(json_line + "\n")

                sleep(30) if model.model_name.endswith("pro") else None

            sleep(10) if model.model_name.endswith("flash") else None


if __name__ == "__main__":
    model_name = "gemini-1.5-pro"
    # model_name = 'gemini-1.5-flash'
    method = "zero-shot"  #'few-shot'
    # model = initial_google_genai(model_name,
    #                              max_output_tokens=2000,
    #                              stop_sequences=['DONE!', '\n\n'])

    MATRES_DATA_PATH = Path("../../data")
    TRC_RAW_PATH = MATRES_DATA_PATH / "TRC"
    results_path = (
        TRC_RAW_PATH / "llm_response" / method / f"platinum-test-{model_name}.jsonl"
    )
    PLATINUM_RAW = MATRES_DATA_PATH / "MATRES" / "raw" / "TBAQ-cleaned" / "te3-platinum"
    parsed_response_path = (
        TRC_RAW_PATH
        / "parsed_responses"
        / method
        / f"platinum-test-results-{model_name}.csv"
    )

    # generate_responses(model,
    #                    platinum_text_prepared_path=TRC_RAW_PATH / 'raw_text' / 'platinum_text_prepared.json',
    #                    prompt_path=TRC_RAW_PATH / 'prompts' / method / 'graph-generation-v1.txt',
    #                    results_path=results_path)

    all_parsed_response_df = pd.DataFrame(
        columns=[
            "docid",
            "verb1",
            "verb2",
            "eiid1",
            "eiid2",
            "relation",
            "unique_id",
            "model_name",
        ]
    ).astype({"unique_id": str})
    results_df = pd.read_json(results_path, lines=True)
    for doc_id, group in results_df.groupby("doc_id"):
        doc = Doc(PLATINUM_RAW / f"{doc_id}.tml")
        for _, row in group.iterrows():
            text = row.text
            response = row.response

            parsed_response_df = prepare_df_from_response(response, doc)
            parsed_response_df["model_name"] = model_name
            all_parsed_response_df = pd.concat(
                [all_parsed_response_df, parsed_response_df], ignore_index=True
            )

    all_parsed_response_df.to_csv(parsed_response_path, index=False)
