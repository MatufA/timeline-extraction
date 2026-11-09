"""Convert native OmniTemp split files into gold and prepared relation records."""

import json
import re
from pathlib import Path

import pandas as pd


_GOLD_COLUMNS = ["docid", "verb1", "verb2", "eiid1", "eiid2", "relation"]
_LABELS = {"BEFORE", "AFTER", "EQUAL", "VAGUE"}
_INVERSE = {"BEFORE": "AFTER", "AFTER": "BEFORE"}


def _event_number(value: object, context: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise ValueError(f"{context}: event ID must be a nonnegative integer or numeric string")


def _mentions(document: dict, tokens: list[str], context: str) -> dict[int, tuple[int, int, str]]:
    mentions = document.get("allMentions")
    if not isinstance(mentions, list):
        raise ValueError(f"{context}: allMentions must be a list")

    events = {}
    occupied = set()
    for index, mention in enumerate(mentions):
        location = f"{context}: allMentions[{index}]"
        if not isinstance(mention, dict):
            raise ValueError(f"{location}: mention must be an object")
        event_id = _event_number(mention.get("m_id"), location)
        if event_id in events:
            raise ValueError(f"{location}: duplicate event ID {event_id}")
        span = mention.get("tokens_ids")
        if not isinstance(span, list) or not span:
            raise ValueError(f"{location}: tokens_ids must be a nonempty list")
        if any(
            not isinstance(position, int)
            or isinstance(position, bool)
            or not 0 <= position < len(tokens)
            for position in span
        ):
            raise ValueError(f"{location}: token span indices must be in-range integers")
        if any(right != left + 1 for left, right in zip(span, span[1:])):
            raise ValueError(f"{location}: token span must be ordered and contiguous")
        if occupied.intersection(span):
            raise ValueError(f"{location}: overlapping event spans are unsupported")
        occupied.update(span)
        # Token indices, not the redundant mention surface/eventIndex, define identity.
        start, end = span[0], span[-1]
        events[event_id] = (start, end, " ".join(tokens[start : end + 1]))
    return events


def _relations(document: dict, events: dict, context: str) -> dict[tuple[int, int], str]:
    pairs = document.get("allPairs")
    if not isinstance(pairs, list):
        raise ValueError(f"{context}: allPairs must be a list")

    relations = {}
    for index, pair in enumerate(pairs):
        location = f"{context}: allPairs[{index}]"
        if not isinstance(pair, dict):
            raise ValueError(f"{location}: relation must be an object")
        first = _event_number(pair.get("_firstId"), location)
        second = _event_number(pair.get("_secondId"), location)
        if first not in events or second not in events:
            raise ValueError(f"{location}: relation references an unknown event ID")
        if first == second:
            raise ValueError(f"{location}: self-relations are unsupported")
        label = pair.get("_relation")
        if not isinstance(label, str) or label.upper() not in _LABELS:
            raise ValueError(f"{location}: unsupported relation label {label!r}")
        label = label.upper()
        if first > second:
            first, second = second, first
            label = _INVERSE.get(label, label)
        key = (first, second)
        if key in relations and relations[key] != label:
            raise ValueError(f"{location}: conflicting annotations for ei{first}, ei{second}")
        relations[key] = label
    return relations


def _render(tokens: list[str], events: dict, targets: set[int]) -> str:
    marked = tokens.copy()
    for event_id in targets:
        start, end, _ = events[event_id]
        marked[start] = f"[EVENT{event_id}]{marked[start]}"
        marked[end] = f"{marked[end]}[/EVENT{event_id}]"
    return " ".join(marked)


def prepare_omnitemp(input_dir: Path) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Read native one-document JSON files without changing or writing any files.

    Files and canonical numeric event pairs are sorted deterministically. Text is
    reconstructed by joining native tokens with spaces; punctuation and complete
    contiguous event spans are preserved. Only explicitly annotated pairs are
    emitted. Discontiguous/overlapping spans and invalid annotations fail rather
    than losing event identities or silently inventing relations.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"OmniTemp input directory does not exist: {input_dir}")
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No OmniTemp JSON documents found in {input_dir}")

    rows = []
    prepared = {"pair": [], "multi": []}
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: cannot read OmniTemp JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"{path}: document must be an object")
        tokens = document.get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(token, str) or not token.strip() for token in tokens)
        ):
            raise ValueError(f"{path}: tokens must be a nonempty list of nonempty strings")
        events = _mentions(document, tokens, str(path))
        relations = _relations(document, events, str(path))
        if not relations:
            continue

        doc_id = path.stem
        doc_pairs = []
        labels = []
        targets = set()
        for (first, second), label in sorted(relations.items()):
            pair = [f"ei{first}", f"ei{second}"]
            rows.append(
                [doc_id, events[first][2], events[second][2], *pair, label]
            )
            prepared["pair"].append(
                {
                    "text": _render(tokens, events, {first, second}),
                    "relations": [pair.copy()],
                    "doc_id": doc_id,
                    "true_label": label.lower(),
                }
            )
            doc_pairs.append(pair)
            labels.append(label)
            targets.update((first, second))
        prepared["multi"].append(
            {
                "text": _render(tokens, events, targets),
                "relations": doc_pairs,
                "doc_id": doc_id,
                "true_label": labels,
            }
        )
    if not rows:
        raise ValueError(f"No usable OmniTemp annotated pairs found in {input_dir}")
    return pd.DataFrame(rows, columns=_GOLD_COLUMNS), prepared
