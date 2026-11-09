import re
import abc
import json
import hashlib
from copy import deepcopy
from typing import List, Type, Optional, Dict, Any, Union

from time import sleep
from pathlib import Path
from tqdm.auto import tqdm

from timeline_extraction.prompts.Prompt import Prompt


def parse_dot_graph(dot_graph: str) -> List[Dict[str, str]]:
    """Extract temporal edges from either EVENT, ei, or e DOT identifiers."""
    pattern = r'"?((?:EVENT|ei|e)\d+)"?\s*->\s*"?((?:EVENT|ei|e)\d+)"?\s*\[label\s*=\s*"?(\w+)"?\s*\]'
    return [
        {
            "event1": f"EVENT{re.search(r'\d+$', e1).group()}",
            "event2": f"EVENT{re.search(r'\d+$', e2).group()}",
            "relation": relation,
        }
        for e1, e2, relation in re.findall(pattern, dot_graph, flags=re.IGNORECASE)
    ]


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
    ) -> Dict[str, Any]:
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
    """Parse JSON or DOT relations, preserving direction when canonicalizing IDs."""

    @staticmethod
    def _event_number(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+$", value.strip())
            if match:
                return int(match.group())
        return None

    def __call__(
        self, model_response: Dict[str, Any], *args, **kwargs
    ) -> List[Dict[str, Any]]:
        response = model_response.get("response", model_response.get("content", ""))
        if isinstance(response, str):
            content = response.strip()
            content = re.sub(r"^```(?:json|dot)?\s*|\s*```$", "", content)
            try:
                response = json.loads(content)
            except json.JSONDecodeError:
                response = parse_dot_graph(content)
                if not response:
                    # Some models emit a sequence of JSON objects instead of an array.
                    response = []
                    decoder = json.JSONDecoder()
                    while content:
                        try:
                            record, end = decoder.raw_decode(content)
                        except json.JSONDecodeError:
                            break
                        response.append(record)
                        content = content[end:].lstrip(" ,\n\r\t")
        if isinstance(response, dict):
            response = [response]
        if not isinstance(response, list):
            return []

        records = []
        seen = set()
        for record in response:
            if not isinstance(record, dict):
                continue
            e1 = self._event_number(record.get("event1"))
            e2 = self._event_number(record.get("event2"))
            relation = record.get("relation")
            if e1 is None or e2 is None or e1 == e2 or not isinstance(relation, str):
                continue
            relation = relation.strip().upper()
            if relation not in {"BEFORE", "AFTER", "EQUAL", "VAGUE"}:
                continue
            if e1 > e2:
                e1, e2 = e2, e1
                relation = {"BEFORE": "AFTER", "AFTER": "BEFORE"}.get(relation, relation)
            key = (e1, e2, relation)
            if key in seen:
                continue
            seen.add(key)
            records.append({**record, "event1": f"ei{e1}", "event2": f"ei{e2}", "relation": relation})
        return records


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
        if n_trails < 1:
            raise ValueError("n_trails must be positive")
        self.model = model
        self.each_trail = each_trail
        self.each_doc = each_doc
        self.n_trails = n_trails
        self.parser = parser()

    @abc.abstractmethod
    def generate_response(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Any]:
        """Generate a response from the model for the given prompt.

        Args:
            prompt: The input prompt for the model

        Returns:
            Native responses, one for each trial
        """
        pass

    @abc.abstractmethod
    def prepare_response(self, response: Any) -> List[Dict[str, Any]]:
        """Prepare and format the raw model response.

        Args:
            response: Raw response from the model

        Returns:
            Independent normalized trial dictionaries including raw content
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
        response_generator=None,
    ) -> List[Dict[str, Any]]:
        """Generate responses for all records in the text file.

        Args:
            text_path: Path to the input text file
            results_path: Path to save the results
            prompt_template: Prompt template to use
            prompt_params: Parameters for the prompt template
            overwrite: Whether to overwrite existing results
            checkpoint_results: Path to save checkpoint results
            response_generator: Optional instrumented generation callable returning normalized trials

        Returns:
            List of generated responses
        """
        if prompt_params is None:
            prompt_params = ["text"]
        with text_path.open() as file:
            records = json.load(file)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_results:
            checkpoint_results.parent.mkdir(parents=True, exist_ok=True)

        completed = {}
        saved_rows = {}
        if not overwrite:
            if checkpoint_results and checkpoint_results.exists():
                with checkpoint_results.open() as file:
                    for line in file:
                        if not line.strip():
                            continue
                        response = json.loads(line)
                        if "request_key" in response:
                            completed.setdefault(response["request_key"], {})[int(response["trail"])] = response
            if results_path.exists():
                with results_path.open() as file:
                    for line in file:
                        if not line.strip():
                            continue
                        response = json.loads(line)
                        if "request_key" in response:
                            saved_rows.setdefault(response["request_key"], {}).setdefault(
                                int(response["trail"]), []
                            ).append(response)

        checkpoint_mode = "w" if overwrite else "a"
        if checkpoint_results and overwrite:
            checkpoint_results.write_text("")
        results = []
        for index, record in enumerate(tqdm(records, desc="Text evaluation", position=0, leave=True)):
            prompt = prompt_template.generate_dict_prompt(
                **{parameter: record[parameter] for parameter in prompt_params}
            )
            identity = json.dumps(
                {"index": index, "record": record, "prompt": prompt},
                sort_keys=True,
                ensure_ascii=False,
            )
            request_key = hashlib.sha256(identity.encode()).hexdigest()
            trials = completed.get(request_key, {})
            prior_rows = saved_rows.get(request_key, {})
            if not all(trial in trials or trial in prior_rows for trial in range(self.n_trails)):
                responses = (
                    response_generator(prompt) if response_generator
                    else self.prepare_response(self.generate_response(prompt))
                )
                for response in responses:
                    trial = int(response["trail"])
                    if trial in trials or trial in prior_rows:
                        continue
                    normalized = {
                        **deepcopy(record),
                        **deepcopy(response),
                        "prompt": prompt,
                        "request_key": request_key,
                    }
                    trials[trial] = normalized
                    if checkpoint_results:
                        with checkpoint_results.open(checkpoint_mode) as file:
                            file.write(json.dumps(normalized) + "\n")
                        checkpoint_mode = "a"
                if self.each_doc:
                    sleep(2)
            for trial in range(self.n_trails):
                if trial in prior_rows and trial not in trials:
                    results.extend(deepcopy(prior_rows[trial]))
                    continue
                if trial not in trials:
                    raise ValueError(f"Provider returned no response for trial {trial}")
                normalized = trials[trial]
                parsed = self.parser(deepcopy(normalized))
                if isinstance(parsed, dict):
                    results.append(parsed)
                else:
                    for relation in parsed:
                        results.append({
                            **deepcopy(normalized),
                            "response": relation["relation"],
                            "relations": [[relation["event1"], relation["event2"]]],
                        })

        with results_path.open("w") as file:
            for result in results:
                file.write(json.dumps(result) + "\n")
        return results
