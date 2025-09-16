import re
import json
import logging
from typing import Optional, Union, Literal
from itertools import combinations

import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import networkx as nx

from timeline_extraction.graph import Graph
from timeline_extraction.utils import (
    normalize_event_id,
    create_unique_id,
    normalize_relation,
    load_csv,
)


def replace_eid(text, exclude_ids):
    pattern = r"ei(\d+):(\w+)"

    def replace_func(match):
        ei_id = match.group(1)
        word = match.group(2)

        if f"ei{ei_id}" in exclude_ids:
            return f"[EVENT{ei_id}]{word}[/EVENT{ei_id}]"
        else:
            return word

    # Use re.sub with a replacement function
    result = re.sub(pattern, replace_func, text)
    return result


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load temporal relation data from file.

    Args:
        path: Path to the data file

    Returns:
        DataFrame with temporal relations
    """
    try:
        df = load_csv(
            path,
            sep="\t",
            header=None,
            names=["docid", "verb1", "verb2", "eiid1", "eiid2", "relation"],
        )
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

    # Normalize event IDs
    df.eiid1 = df.eiid1.astype(str).apply(normalize_event_id)
    df.eiid2 = df.eiid2.astype(str).apply(normalize_event_id)

    # Normalize relations
    df["relation"] = df["relation"].apply(normalize_relation)

    # Handle AFTER relations by swapping events and converting to BEFORE
    mask = df["relation"] == "AFTER"
    df.loc[mask, ["verb1", "verb2"]] = df.loc[mask, ["verb2", "verb1"]].values
    df.loc[mask, ["eiid1", "eiid2"]] = df.loc[mask, ["eiid2", "eiid1"]].values
    df["label"] = df["relation"].copy()
    df["relation"] = df["relation"].replace("AFTER", "BEFORE")

    # Create unique IDs for event pairs
    df["unique_id"] = df.apply(
        lambda row: create_unique_id(row["eiid1"], row["eiid2"]), axis=1
    )

    return df


def generate_all_comb_training_dataset(
    docs_dir: Union[str, Path],
    output_path: Path,
    window: int = 2,
    is_full_text: bool = False,
    prev_relations_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    docs = [Doc(path) for path in Path(docs_dir).glob("*.tml")]
    ei_regex = r"(ei\d+):\w+\s*"

    if prev_relations_df is None:
        doc_relations = []
    else:
        doc_relations = [
            {
                "text": replace_eid(row.text, exclude_ids=[row.eiid1, row.eiid2]),
                "relations": [(row.eiid1, row.eiid2)],
                "doc_id": row.docid,
            }
            for _, row in prev_relations_df.iterrows()
        ]

    for doc in docs:
        full_text = doc.get_text()
        lines = full_text.split("\n\n")

        event_dict = {}
        for idx, line in enumerate(lines):
            ids_groups = re.findall(ei_regex, line)
            if ids_groups:
                indexes = np.repeat(idx, len(ids_groups))
                event_dict.update(dict(zip(ids_groups, indexes)))

        unique_ids = (
            set(
                prev_relations_df[prev_relations_df.docid == doc.docid].unique_id.values
            )
            if prev_relations_df is not None
            else set()
        )
        lines_size = len(lines)
        for idx, line in enumerate(lines):
            max_idx = min(lines_size, idx + window) + 1
            window_text = "\n".join(lines[idx:max_idx])
            ids_groups = re.findall(ei_regex, window_text)

            if ids_groups:
                rel_combs = combinations(ids_groups, 2)
                for comb in rel_combs:
                    comb_key = "-".join(sorted(comb))
                    if comb_key not in unique_ids:
                        min_idx = min(event_dict[comb[0]], event_dict[comb[1]])
                        max_idx = max(event_dict[comb[0]], event_dict[comb[1]])

                        if is_full_text:
                            train_text = full_text
                        else:
                            train_text = "\n".join(lines[min_idx : max_idx + 1])

                        unique_ids.add(comb_key)
                        doc_relations.append(
                            {
                                "doc_id": doc.docid,
                                "text": replace_eid(train_text, exclude_ids=comb),
                                "relations": [list(comb)],
                            }
                        )
    output_path.write_text(json.dumps(doc_relations, indent=2))


def generate_training_dataset(
    gold_df: pd.DataFrame, docs_dir: Union[str, Path], mode: str
) -> pd.DataFrame:
    docs = [Doc(path) for path in Path(docs_dir).glob("*.tml")]

    doc_relations = []
    for doc in docs:
        doc_relation = gold_df[gold_df["docid"] == doc.docid].copy()
        doc_relation["text"] = pd.NA

        # doc.get_text_with_relation_external()
        lines = doc.get_text().split("\n\n")

        event_dict = {}
        for idx, line in enumerate(lines):
            ids_groups = re.findall(r"(ei\d+):\w+\s*", line)
            if ids_groups:
                indexes = np.repeat(idx, len(ids_groups))
                event_dict.update(dict(zip(ids_groups, indexes)))

        # plain_text_lines = re.sub(r'ei\d+:(\w+)\s*', r'\1 ', doc.get_text()).split('\n\n')
        plain_text_lines = doc.get_text().split("\n\n")

        for idx, row in doc_relation.iterrows():
            if row["eiid1"] not in event_dict:
                logging.warning(f'docid: {doc.docid} and eid: {row["eiid1"]}')
                continue

            if row["eiid2"] not in event_dict:
                logging.warning(f'docid: {doc.docid} and eid: {row["eiid2"]}')
                continue

            e1_index = event_dict[row["eiid1"]]
            e2_index = event_dict[row["eiid2"]]

            if mode == "pair":
                if e1_index != e2_index:
                    min_idx = min(e1_index, e2_index)
                    max_idx = max(e1_index, e2_index)

                    doc_relation.at[idx, "text"] = "\n".join(
                        plain_text_lines[min_idx : max_idx + 1]
                    )
                else:
                    doc_relation.at[idx, "text"] = plain_text_lines[e1_index]
            elif mode == "multi":
                doc_relation.at[idx, "text"] = "\n".join(plain_text_lines)

        doc_relations.append(doc_relation)

    return pd.concat(doc_relations, ignore_index=True).dropna(subset="text")


def prepare_as_jsonl(
    df: pd.DataFrame, output_path: Path, mode: Literal["multi", "pair"]
) -> None:

    data = []

    if mode == "pair":
        for idx, row in df.iterrows():
            data.append(
                {
                    "text": replace_eid(row.text, exclude_ids=[row.eiid1, row.eiid2]),
                    "relations": [(row.eiid1, row.eiid2)],
                    "doc_id": row.docid,
                    "true_label": row.label.lower(),
                }
            )
    else:
        for doc_id, group in df.groupby("docid"):
            # relations = '\n'.join(group.loc[:, ['eiid1', 'eiid2']].apply(lambda row: ' [RELATION] '.join(row), axis=1))
            unique_relation_ids = np.unique(
                group.loc[:, ["eiid1", "eiid2"]].to_numpy().flatten()
            )
            data.append(
                {
                    "text": replace_eid(
                        group.iloc[0].text, exclude_ids=unique_relation_ids
                    ),
                    "relations": group.loc[:, ["eiid1", "eiid2"]]
                    .apply(lambda row: tuple(row), axis=1)
                    .values.tolist(),
                    "doc_id": doc_id,
                    "true_label": group.loc[:, "label"].to_numpy().tolist(),
                }
            )

    output_path.write_text(json.dumps(data, indent=2))


class Doc:
    def __init__(self, path: Path):
        self.path = path
        self.doc = self.parse()
        self.docid = self.doc.find("docid").get_text()
        self.mapping = self.get_mapping()
        self.text = self.get_text()
        self.relations = self.get_relations()

    def parse(self):
        with self.path.open("r") as f:
            data = f.read()
        return BeautifulSoup(data, features="lxml")

    def get_mapping(self):
        eid_mapping = {}
        for d in self.doc.find_all("makeinstance"):
            if d.get("eventid") not in eid_mapping:
                eid_mapping[d.get("eventid")] = d.get("eiid")
        return eid_mapping

    def get_text(self):
        text = self.doc.find(name="text")
        [
            d.replace_with(
                f'{self.mapping[d.get("eid")] if d.get("eid") in self.mapping else d.get("eid")}:{d.get_text()}'
            )
            for d in text.find_all("event")
        ]
        [d.replace_with(d.get_text()) for d in text.find_all("timex3")]
        return text.get_text().strip()

    def get_relations(self):
        relations = (
            (
                d.get("eventinstanceid"),
                d.get("relatedtoeventinstance"),
                d.get("reltype"),
            )
            for d in self.doc.find_all("tlink")
            if d.get("reltype") in set(["IBEFORE", "BEFORE", "AFTER", "IAFTER"])
        )
        return list(filter(lambda d: d[0] is not None and d[1] is not None, relations))


def generate_transitive_reduction_dataset(
    df: pd.DataFrame, output_path: Path, use_equal: bool = True
) -> pd.DataFrame:
    supported_relation = ["AFTER", "BEFORE"]
    if use_equal:
        supported_relation.append("EQUAL")

    tr_list = []
    for docid, group in df.groupby("docid"):
        graph = Graph(
            use_equal=use_equal,
            supported_relation=supported_relation,
            relation_key="label",
        )
        doc_graph = graph.generate_directed_graph(group)
        tr = nx.transitive_reduction(doc_graph)
        tr_list.extend([e1.split("-") + [e2.split("-")[1]] for e1, e2 in tr.edges])

    tr_df = pd.DataFrame(tr_list, columns=["docid", "eiid1", "eiid2"])
    eiid1_eiid2 = list(zip(tr_df["eiid1"], tr_df["eiid2"]))
    tr_df["unique_id"] = [
        "-".join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2
    ]

    new_df = pd.merge(df, tr_df, on=["docid", "unique_id"], how="inner")
    new_df.to_csv(output_path, index=False)
    return new_df


if __name__ == "__main__":
    DATA_PATH = Path("./data")
    MATRES_PATH = DATA_PATH / "MATRES"
    DOCS_DIRS_PATH = MATRES_PATH / "raw" / "TBAQ-cleaned"

    df = load_data(MATRES_PATH / "platinum.txt")
    # generate_transitive_reduction_dataset(df, output_path=MATRES_PATH / 'platinum_dags.txt', use_equal=False)

    BASE_NAMES = ["te3-platinum"]  #  'AQUAINT', 'TimeBank',
    BASE_DF_PATHS = ["platinum.txt"]  # 'aquaint.txt', 'timebank.txt',
    modes = ["comb"]  # 'pair', 'multi',
    is_full_text = False

    for mode in modes:

        if mode == "pair":
            output = DATA_PATH / "wo-vague" / "trc-prepared-data"
        elif mode == "multi":
            output = DATA_PATH / "wo-vague" / "te-prepared-data"
        elif mode == "comb":
            output = DATA_PATH / "wo-vague" / "trc-all-prepared-data"

        for name, df_name in zip(BASE_NAMES, BASE_DF_PATHS):
            output_file = output / f"{mode}-{name}.csv"

            context_ref = "full_context" if is_full_text else "minimal_context"
            jsonl_out_name = (
                f"{mode}_{name.lower()}_{context_ref}_text_w_relations_prepared.json"
            )
            jsonl_out_path = DATA_PATH / "TRC" / "raw_text" / jsonl_out_name
            overwrite = True

            if not output_file.exists() or overwrite:
                gold_df = load_data(MATRES_PATH / df_name)
                # gold_df = gold_df[gold_df.label != 'VAGUE']

                df = generate_training_dataset(
                    gold_df, docs_dir=DOCS_DIRS_PATH / name, mode="pair"
                )
                output.mkdir(exist_ok=True, parents=True)
                df.to_csv(output_file, index=False)
            else:
                df = pd.read_csv(output_file, header=0)
                print(f"Data already exists for {name}. Skipping.")

            prepare_as_jsonl(df, output_path=jsonl_out_path, mode=mode)

            if mode == "comb":

                df_comb = generate_all_comb_training_dataset(
                    docs_dir=DOCS_DIRS_PATH / name,
                    output_path=jsonl_out_path,
                    window=2,
                    is_full_text=is_full_text,
                    prev_relations_df=df,
                )
