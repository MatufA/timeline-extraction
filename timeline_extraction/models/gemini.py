from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from time import sleep
from typing import Any, Dict, List, Optional, Type

from google import genai
from google.genai import types
from tqdm.auto import tqdm

from timeline_extraction.models.LLModel import LLModel, LabelParser, Parser
from timeline_extraction.models._responses import chat_messages
from timeline_extraction.models._reasoning import reasoning_options
from timeline_extraction.prompts.Prompt import Prompt
# Keep all Decision Model question wording, options, and criteria in one place.
DECISION_QUESTION_ID = "temporal_relation"
DECISION_QUESTION_TEMPLATE = (
    "Classify the chronological order of {event_1} relative to {event_2} in the text."
)
DECISION_CRITERIA_TEMPLATES = {
    "before": "{event_1} happened or started before {event_2}.",
    "after": "{event_1} happened or started after {event_2} ({event_2} happened first).",
    "equal": "{event_1} and {event_2} happened at the same time.",
}
DECISION_CRITERIA_VAGUE_TEMPLATE = (
    "It is impossible to know from the text whether {event_1} happened "
    "before, after, or at the same time as {event_2}."
)
DECISION_FEW_SHOT_PREFIX = (
    "Examples:\n"
    '1. "Suspected bombs [EVENT1]exploded[/EVENT1] outside the U.S. embassies Friday, '
    '[EVENT2]killing[/EVENT2] dozens of people, witnesses said." '
    "([EVENT1]exploded[/EVENT1] relative to [EVENT2]killing[/EVENT2]: before)\n"
    '2. "Suspected bombs exploded outside the U.S. embassies Friday, killing dozens '
    "of people, witnesses [EVENT3]said[/EVENT3]. The American ambassador to Kenya was "
    'among hundreds [EVENT12]injured[/EVENT12], a local TV said." '
    "([EVENT3]said[/EVENT3] relative to [EVENT12]injured[/EVENT12]: after)\n\n"
    "Text:\n"
)


def _is_decision_model(model_name: str) -> bool:
    return model_name.lower().removeprefix("models/").startswith("gemini-decision")


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
        self.is_decision_model = _is_decision_model(model_name)
        if self.is_decision_model and not issubclass(parser, LabelParser):
            raise ValueError(
                f"{model_name} is a decision model and only supports pairwise classification (pair or comb mode)"
            )
        self.reasoning_options = reasoning_options("google", model_name, reasoning_effort)
        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        extra_keys = [
            k.strip()
            for k in (os.environ.get("GEMINI_API_KEYS") or "").split(",")
            if k.strip()
        ]
        all_keys = list(dict.fromkeys([k for k in [resolved_key, *extra_keys] if k]))
        if not all_keys:
            all_keys = [resolved_key]
        client_kwargs = {"http_options": {"api_version": "v1beta"}} if self.is_decision_model else {}
        self._clients = [genai.Client(api_key=k, **client_kwargs) for k in all_keys]
        self._client_idx = 0
        super().__init__(
            self._clients[0],
            each_trail=False,
            each_doc=not self.is_decision_model,
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

    @staticmethod
    def _event_span(passage: str, event_id: str) -> str:
        match = re.search(rf"\[EVENT{event_id}\].*?\[/EVENT{event_id}\]", passage, re.DOTALL)
        return match.group(0) if match else f"EVENT{event_id}"

    @classmethod
    def _build_question(cls, passage: str, first_id: str, second_id: str, use_vague: bool) -> Dict[str, Any]:
        event_1 = cls._event_span(passage, first_id)
        event_2 = cls._event_span(passage, second_id)
        criteria = {
            label: template.format(event_1=event_1, event_2=event_2)
            for label, template in DECISION_CRITERIA_TEMPLATES.items()
        }
        if use_vague:
            criteria["vague"] = DECISION_CRITERIA_VAGUE_TEMPLATE.format(
                event_1=event_1, event_2=event_2
            )
        return {
            "type": "choice",
            "instructions": DECISION_QUESTION_TEMPLATE.format(
                event_1=event_1, event_2=event_2
            ),
            "criteria": criteria,
        }

    @classmethod
    def _decision_request(cls, prompt):
        if isinstance(prompt, dict) and "input" in prompt and "questions" in prompt:
            return prompt["input"], prompt["questions"]
        contents, instructions = cls._contents(prompt)
        user_text = "\n\n".join(
            part.text
            for content in contents
            for part in (content.parts or [])
            if getattr(part, "text", None)
        ).strip()
        if not user_text:
            raise ValueError("Gemini decision prompts require non-empty user content")

        passage_match = re.search(
            r"Text for Analysis:\s*(.*?)\s*---",
            user_text,
            re.DOTALL,
        )
        if passage_match:
            passage = passage_match.group(1).strip()
        else:
            passage = re.sub(
                r"\s*(?:Classify\s+EVENT\d+\s+relative\s+to\s+EVENT\d+.*|in one word\s*-->.*)$",
                "",
                user_text,
                flags=re.DOTALL,
            ).strip()
        if not passage:
            raise ValueError("Gemini decision prompt contains no passage text")

        pair_match = re.search(
            r"Classify\s+EVENT(\d+)\s+relative\s+to\s+EVENT(\d+)",
            user_text,
        )
        if pair_match:
            first_id, second_id = pair_match.group(1), pair_match.group(2)
        else:
            seen_ids = list(dict.fromkeys(re.findall(r"\[EVENT(\d+)\]", passage)))
            if len(seen_ids) < 2:
                raise ValueError(
                    "Gemini decision prompt must identify two target events"
                )
            first_id, second_id = seen_ids[0], seen_ids[1]

        input_text = (
            f"{DECISION_FEW_SHOT_PREFIX}{passage}"
            if instructions and "examples:" in instructions.lower()
            else passage
        )
        use_vague = True if not instructions else ("vague" in instructions.lower())
        questions = {
            DECISION_QUESTION_ID: cls._build_question(
                passage, first_id, second_id, use_vague
            )
        }
        return input_text, questions

    def _call_decisions_api(self, model_id: str, input_text: str, questions: Dict[str, Any]):
        clients = getattr(self, "_clients", None) or [self.model]
        last_error = None
        for attempt in range(max(len(clients) * 4, 6)):
            idx = getattr(self, "_client_idx", 0) % len(clients)
            self._client_idx = idx + 1
            client = clients[idx]
            try:
                return client.decisions.create(
                    model=model_id,
                    input=input_text,
                    questions=questions,
                    api_version="v1beta",
                )
            except Exception as error:
                last_error = error
                msg = str(error)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "Too Many Requests" in msg:
                    sleep(min(2.0 * (attempt + 1), 15.0))
                    continue
                raise
        raise last_error

    def generate_response(self, prompt):
        if getattr(self, "is_decision_model", False) or _is_decision_model(getattr(self, "model_name", "")):
            input_text, questions = self._decision_request(prompt)
            model_id = self.model_name.removeprefix("models/")
            return [
                self._call_decisions_api(model_id, input_text, questions)
                for _ in range(self.n_trails)
            ]
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

    @staticmethod
    def _serialize_decision_answer(answer) -> tuple[str, str]:
        choice = getattr(answer, "choice", None)
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError("Gemini decision response returned no choice")
        probabilities = dict(getattr(answer, "probabilities", None) or {})
        confidence = getattr(answer, "confidence", None)
        content = json.dumps(
            {
                "choice": choice.strip(),
                "confidence": confidence,
                "probabilities": probabilities,
            },
            sort_keys=True,
        )
        return choice.strip(), content

    def prepare_response(self, responses):
        results = []
        for trail, response in enumerate(responses):
            if hasattr(response, "answers") and not hasattr(response, "candidates"):
                if getattr(response, "prompt_feedback", None) is not None:
                    raise ValueError(
                        f"Gemini decision request blocked: {response.prompt_feedback}"
                    )
                answers = response.answers or {}
                if not answers:
                    raise ValueError(
                        f"Gemini decision response missing {DECISION_QUESTION_ID!r} answer"
                    )
                usage = getattr(response, "usage", None)
                if DECISION_QUESTION_ID in answers and len(answers) == 1:
                    choice, content = self._serialize_decision_answer(
                        answers[DECISION_QUESTION_ID]
                    )
                    results.append(
                        {
                            "trail": trail,
                            "response": choice,
                            "content": content,
                            "prompt_tokens": getattr(usage, "total_input_tokens", None),
                            "completion_tokens": getattr(usage, "total_output_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
                    )
                else:
                    if DECISION_QUESTION_ID not in answers and not any(
                        k.startswith("pair_") for k in answers
                    ):
                        raise ValueError(
                            f"Gemini decision response missing {DECISION_QUESTION_ID!r} answer"
                        )
                    serialized = {
                        qid: dict(zip(("response", "content"), self._serialize_decision_answer(ans)))
                        for qid, ans in answers.items()
                    }
                    first_key = next(iter(serialized))
                    results.append(
                        {
                            "trail": trail,
                            "response": serialized[first_key]["response"],
                            "content": json.dumps(serialized, sort_keys=True),
                            "answers": serialized,
                            "prompt_tokens": getattr(usage, "total_input_tokens", None),
                            "completion_tokens": getattr(usage, "total_output_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
                    )
                continue
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

    @staticmethod
    def _unmarked_and_spans(marked_text: str):
        tag_re = re.compile(r"\[EVENT(\d+)\](.*?)\[/EVENT\1\]", re.DOTALL)
        spans = {}
        out = []
        pos = 0
        last = 0
        for match in tag_re.finditer(marked_text):
            prefix = marked_text[last : match.start()]
            out.append(prefix)
            pos += len(prefix)
            inner = match.group(2)
            eid = match.group(1)
            spans[(pos, pos + len(inner), eid)] = inner
            out.append(inner)
            pos += len(inner)
            last = match.end()
        out.append(marked_text[last:])
        return "".join(out), spans

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
        """Bundle questions sharing a passage into one Decisions API request."""
        if not (getattr(self, "is_decision_model", False) or _is_decision_model(getattr(self, "model_name", ""))):
            return super().generate_responses(
                text_path=text_path,
                results_path=results_path,
                prompt_template=prompt_template,
                prompt_params=prompt_params,
                overwrite=overwrite,
                checkpoint_results=checkpoint_results,
                response_generator=response_generator,
            )
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

        prepared_items = []
        by_doc = defaultdict(list)
        for index, record in enumerate(records):
            prompt = prompt_template.generate_dict_prompt(
                **{parameter: record[parameter] for parameter in prompt_params}
            )
            identity = json.dumps(
                {"index": index, "record": record, "prompt": prompt},
                sort_keys=True,
                ensure_ascii=False,
            )
            request_key = hashlib.sha256(identity.encode()).hexdigest()
            trials = completed.setdefault(request_key, {})
            prior_rows = saved_rows.get(request_key, {})
            item = (index, record, prompt, request_key, trials, prior_rows)
            prepared_items.append(item)
            if not all(trial in trials or trial in prior_rows for trial in range(self.n_trails)):
                unmarked, spans = self._unmarked_and_spans(record["text"])
                by_doc[record.get("doc_id", "")].append((item, unmarked, spans))

        doc_passages = []
        for doc_id, doc_items in by_doc.items():
            paras = []
            for _, unmarked, _ in doc_items:
                for line in unmarked.split("\n"):
                    if line not in paras:
                        paras.append(line)
            full_unmarked = "\n".join(paras)
            containers = [full_unmarked]
            for _, unmarked, _ in doc_items:
                if full_unmarked.find(unmarked) == -1 and not any(unmarked in c for c in containers):
                    containers.append(unmarked)
            remaining = list(doc_items)
            for container in containers:
                c_spans = {}
                c_group = []
                next_remaining = []
                for entry in remaining:
                    item, unmarked, spans = entry
                    offset = container.find(unmarked)
                    if offset != -1:
                        c_group.append(item)
                        for (start, end, eid), inner in spans.items():
                            c_spans[(offset + start, offset + end, eid)] = inner
                    else:
                        next_remaining.append(entry)
                remaining = next_remaining
                if not c_group:
                    continue
                merged_passage = container
                for start, end, eid in sorted(c_spans, key=lambda x: x[0], reverse=True):
                    merged_passage = (
                        f"{merged_passage[:start]}[EVENT{eid}]"
                        f"{merged_passage[start:end]}[/EVENT{eid}]{merged_passage[end:]}"
                    )
                doc_passages.append((doc_id, merged_passage, c_group))

        # Pack up to 1,000 questions into a single Decisions API request.
        batches = []
        cur_docs = []
        cur_count = 0
        for doc_id, merged_passage, c_group in doc_passages:
            idx_offset = 0
            while idx_offset < len(c_group):
                room = 1000 - cur_count
                take = c_group[idx_offset : idx_offset + room]
                cur_docs.append((doc_id, merged_passage, take))
                cur_count += len(take)
                idx_offset += len(take)
                if cur_count >= 1000:
                    batches.append(cur_docs)
                    cur_docs = []
                    cur_count = 0
        if cur_docs:
            batches.append(cur_docs)

        use_vague = getattr(prompt_template, "use_vague", True)
        use_few_shot = getattr(prompt_template, "use_few_shot", False)
        for packed_docs in tqdm(batches, desc="Bundled decision requests", position=0, leave=True):
            multi_doc = len(packed_docs) > 1
            if multi_doc:
                seen_doc_sections = {}
                for doc_id, merged_passage, _ in packed_docs:
                    seen_doc_sections.setdefault(doc_id, f"[Document {doc_id}]\n{merged_passage}")
                combined_passage = "\n\n".join(seen_doc_sections.values())
            else:
                combined_passage = packed_docs[0][1]
            input_text = (
                f"{DECISION_FEW_SHOT_PREFIX}{combined_passage}"
                if use_few_shot
                else combined_passage
            )
            questions = {}
            group = []
            for doc_id, merged_passage, sub_group in packed_docs:
                for item in sub_group:
                    index, record, *_ = item
                    group.append(item)
                    first, second = record["relations"][0]
                    first_id = re.search(r"\d+$", str(first)).group()
                    second_id = re.search(r"\d+$", str(second)).group()
                    q_obj = self._build_question(
                        merged_passage, first_id, second_id, use_vague
                    )
                    if multi_doc and doc_id:
                        q_obj["instructions"] = f"In [Document {doc_id}], {q_obj['instructions'][0].lower()}{q_obj['instructions'][1:]}"
                    questions[f"pair_{index}"] = q_obj
            batched_prompt = {"input": input_text, "questions": questions}
            batch_responses = (
                response_generator(batched_prompt)
                if response_generator
                else self.prepare_response(self.generate_response(batched_prompt))
            )
            for batch_row in batch_responses:
                trial = int(batch_row["trail"])
                answers_map = batch_row.get("answers")
                for pos, (index, record, prompt, request_key, trials, prior_rows) in enumerate(group):
                    if trial in trials or trial in prior_rows:
                        continue
                    qid = f"pair_{index}"
                    if answers_map and qid in answers_map:
                        ans_entry = answers_map[qid]
                        resp_label = ans_entry["response"]
                        resp_content = ans_entry["content"]
                    else:
                        resp_label = batch_row["response"]
                        resp_content = batch_row["content"]
                    normalized = {
                        **deepcopy(record),
                        "trail": trial,
                        "response": resp_label,
                        "content": resp_content,
                        "prompt_tokens": batch_row.get("prompt_tokens") if pos == 0 else 0,
                        "completion_tokens": batch_row.get("completion_tokens") if pos == 0 else 0,
                        "total_tokens": batch_row.get("total_tokens") if pos == 0 else 0,
                        "prompt": prompt,
                        "request_key": request_key,
                    }
                    trials[trial] = normalized
                    if checkpoint_results:
                        with checkpoint_results.open(checkpoint_mode) as file:
                            file.write(json.dumps(normalized) + "\n")
                        checkpoint_mode = "a"

        results = []
        for index, record, prompt, request_key, trials, prior_rows in prepared_items:
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
