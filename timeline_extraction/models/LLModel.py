import re
import abc
import json
from typing import List, Type, Optional, Dict, Any

import pandas as pd
from time import sleep
from pathlib import Path
from tqdm.auto import tqdm

from timeline_extraction.prompts.Prompt import Prompt


def parse_dot_graph(dot_graph: str) -> List[Dict[str, str]]:
    # Extract edges from the DOT graph
    edge_pattern = r'"?(EVENT\d+)"?\s*->\s*"?(EVENT\d+)"?\s*\[label="?([\w]*)"?\]?'
    matches = re.findall(edge_pattern, dot_graph)

    # Create a list of dictionaries from the matches
    relations = [
        {"event1": match[0], "event2": match[1], "relation": match[2]}
        for match in matches
    ]

    if not relations:
        edge_pattern = r'"?(e\d+)"?\s*->\s*"?(e\d+)"?\s*\[label=("?[\w]*)"?\]?'
        matches = re.findall(edge_pattern, dot_graph)

        relations = [
            {
                "event1": match[0].replace("e", "EVENT"),
                "event2": match[1].replace("e", "EVENT"),
                "relation": match[2],
            }
            for match in matches
        ]

    return relations


class Parser:
    """Abstract base class for parsing model responses."""

    @abc.abstractmethod
    def __call__(self, model_response: Dict[str, Any], *args, **kwargs) -> Any:
        """Parse model response and extract relevant information."""
        pass


class NoParser(Parser):
    """Parser that returns the raw model response without processing."""

    def __call__(
        self, model_response: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, Any]:
        """Return the model response unchanged."""
        return model_response


class LabelParser(Parser):
    """Parser for extracting temporal relation labels from model responses."""

    def __call__(
        self, model_response: Dict[str, Any], *args, **kwargs
    ) -> Optional[str]:
        start_pattern = r"^(before|after|equal|vague)\b"
        end_pattern = r"\b(before|after|equal|vague)(?:[.!?]|\s*$)"
        if start_label_match := re.search(
            start_pattern, model_response["response"].lower()
        ):
            plabel = start_label_match.group(1)
        elif end_label_match := re.search(
            end_pattern, model_response["response"].lower()
        ):
            plabel = end_label_match.group(1)
        else:
            plabel = model_response["response"]

        model_response["response"] = plabel
        return model_response


class JsonParser(Parser):
    """Parser for extracting JSON-formatted temporal relations from model responses."""

    def __call__(
        self, model_response: Dict[str, Any], *args, **kwargs
    ) -> List[Dict[str, Any]]:
        if isinstance(model_response["response"], list) and isinstance(
            model_response["response"][0], dict
        ):
            formated_response = model_response["response"]
        else:
            content = (
                model_response["content"].strip().replace("[", "").replace("]", "")
            )
            content = f"[{content}]"

            try:
                formated_response = json.loads(content.replace("\n", ""))
            except json.JSONDecodeError:
                try:
                    formated_response = json.loads(
                        model_response["content"].replace(",\n", "").replace("\n", "")
                    )
                except json.JSONDecodeError:
                    formated_response = []
                    for rec in (
                        model_response["content"]
                        .replace("\n", "")
                        .replace("},{", "}${")
                        .split("$")
                    ):
                        try:
                            formated_response.append(json.loads(rec))
                        except json.JSONDecodeError:
                            pass

        if isinstance(formated_response, list):
            new_formated_response = []
            for record in formated_response:
                try:
                    e1, e2 = record["event1"], record["event2"]
                except KeyError:
                    e1_key, e2_key = sorted(
                        [k for k in record if "event" in k],
                        key=lambda e: int(re.search(r"\d+$", e).group()),
                    )
                    e1, e2 = record[e1_key], record[e2_key]

                # validate events are not None
                if not e1 and not e2:
                    continue

                # parse and fix events formate
                if not isinstance(e1, int) and not isinstance(e2, int):
                    events = [e1, e2]
                    formated_events = []
                    for e in events:
                        if number_from_end := re.search(r"\d+$", e):
                            formated_events.append(int(number_from_end.group()))
                        elif number_from_start := re.search(r"^\d+", e):
                            formated_events.append(int(number_from_start.group()))
                        else:
                            print(f"unsupported record: {record}")
                else:
                    formated_events = [e1, e2]

                if formated_events:
                    new_recored = record.copy()
                    new_recored["event1"], new_recored["event2"] = (
                        f"ei{min(formated_events)}",
                        f"ei{max(formated_events)}",
                    )
                    new_formated_response.append(new_recored)

            df = pd.DataFrame.from_dict(new_formated_response)
            df = df.drop_duplicates()
            formated_response = df.to_dict(orient="records")

        return formated_response


class LLModel(abc.ABC):
    """Abstract base class for Large Language Model implementations."""

    def __init__(
        self,
        model: Any,
        each_trail: bool = False,
        each_doc: bool = True,
        n_trails: int = 5,
        parser: Type[Parser] = LabelParser,
    ) -> None:
        """Initialize the LLM with configuration parameters.

        Args:
            model: The underlying model instance
            each_trail: Whether to run multiple trials for each request
            each_doc: Whether to process each document separately
            n_trails: Number of trials to run
            parser: Parser class for processing model responses
        """
        self.model = model
        self.each_trail = each_trail
        self.each_doc = each_doc
        self.n_trails = n_trails
        self.parser = parser()

    @abc.abstractmethod
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a response from the model for the given prompt.

        Args:
            prompt: The input prompt for the model

        Returns:
            Dictionary containing the model response and metadata
        """
        pass

    @abc.abstractmethod
    def prepare_response(self, response: Any) -> Dict[str, Any]:
        """Prepare and format the raw model response.

        Args:
            response: Raw response from the model

        Returns:
            Formatted response dictionary
        """
        pass

    def generate_responses(
        self,
        text_path: Path,
        results_path: Path,
        prompt_template: Prompt,
        prompt_params: Optional[List[str]] = None,
        overwrite: bool = False,
        checkpoint_results: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate responses for all records in the text file.

        Args:
            text_path: Path to the input text file
            results_path: Path to save the results
            prompt_template: Prompt template to use
            prompt_params: Parameters for the prompt template
            overwrite: Whether to overwrite existing results
            checkpoint_results: Path to save checkpoint results

        Returns:
            List of generated responses
        """
        if prompt_params is None:
            prompt_params = ["text"]

        records = json.load(text_path.open("r"))
        if checkpoint_results:
            with checkpoint_results.open("w") as clean_file:
                clean_file.write("")
        # if 'relations' in prompt_params:
        # prompt_params = ['text']
        # for record in records:
        #     rels = []
        #     for rel in record['relations'].split('\n'):
        #         e1, e2 = rel.split(' [RELATION] ')
        #         e1, e2 = e1.strip(), e2.strip()
        #         # rels.append(json.dumps({"event1": e1, "event2": e2}))
        #         rels.append(f'{e1} {e2}')
        #         new_records.append({
        #             'text': replace_eid(record['text'], exclude_ids=[e1, e2]),
        #             'doc_id': record["doc_id"],
        #             'e1': e1,
        #             'e2': e2
        #             })

        # record['relations'] = '\n'.join(rels)

        # prompt_template = prompt_path.open('r').read()

        if not overwrite and results_path.exists():
            write_mode = "a"
            for record in records:
                record["key"] = (
                    record["doc_id"] + "-" + "-".join(record["relations"][0])
                )

            records_partial_df: pd.DataFrame = pd.read_json(results_path, lines=True)

            if not records_partial_df.empty:
                # doc_iter = records_partial_df.groupby('doc_id')['n_tokens'].count().to_dict()

                # doc_iter = records_partial_df.key.unique()
                # records = (record for record in records if record['doc_id'] not in doc_iter)

                doc_iter = set(
                    records_partial_df.apply(
                        lambda row: row["doc_id"] + "-" + "-".join(row["relations"][0]),
                        axis="columns",
                    )
                )
                records = [
                    record for record in records if record["key"] not in doc_iter
                ]
        else:
            write_mode = "w"

        resluts = []
        with results_path.open(write_mode) as file:
            for record in tqdm(records, desc="Text evaluation", position=0, leave=True):
                prompt = prompt_template.generate_dict_prompt(
                    **{p: record[p] for p in prompt_params}
                )
                response = self.generate_response(prompt)

                if checkpoint_results:
                    with checkpoint_results.open("a") as c_file:
                        for choice in response.choices:
                            choice_result = dict()
                            choice_result["doc_id"] = record["doc_id"]
                            choice_result["prompt"] = prompt
                            choice_result["raw_content"] = choice.message.content
                            if isinstance(self.parser, LabelParser):
                                choice_result["relations"] = choice.message.content
                            else:
                                choice_result["relations"] = parse_dot_graph(
                                    dot_graph=choice.message.content
                                )
                            j_line = json.dumps(choice_result)
                            c_file.write(j_line + "\n")

                for res in self.prepare_response(response):
                    res.update(record)
                    res["prompt"] = prompt
                    if not res:
                        print(f"unable to process res: {response}")
                        continue
                    parsed_response = self.parser(res)
                    if not isinstance(parsed_response, list):
                        parsed_response = [parsed_response]

                    for p_response in parsed_response:
                        if not isinstance(self.parser, LabelParser):
                            res["response"] = p_response["relation"]
                            # if len(res['relations']) > 1:
                            events = [p_response["event1"], p_response["event2"]]
                            res["relations"] = [events]
                        # if isinstance(prompt_template, PairwisePrompt) and not isinstance(res['response'], dict) :
                        #     start_pattern = r'^(before|after|equal|vague)\b'
                        #     end_pattern = r'\b(before|after|equal|vague)(?:[.!?]|\s*$)'
                        #     if start_label_match := re.search(start_pattern, res['response'].lower()):
                        #         plabel = start_label_match.group(1)
                        #     elif end_label_match := re.search(end_pattern, res['response'].lower()):
                        #         plabel = end_label_match.group(1)
                        #     else:
                        #         plabel = res['response']

                        #     res['response'] = plabel

                        json_line = json.dumps(res)
                        file.write(json_line + "\n")
                        resluts.append(res)

                sleep(2) if self.each_doc else None
        return resluts
