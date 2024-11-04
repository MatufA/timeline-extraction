import logging
import re
from typing import Union

import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup


def load_data(paht: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(paht, sep='\t', header=None,
                     names=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2', 'relation'])
    df.eiid1 = 'ei' + df.eiid1.astype(str)
    df.eiid2 = 'ei' + df.eiid2.astype(str)

    mask = df['relation'] == 'after'
    df.loc[mask, ['verb1', 'verb2']] = df.loc[mask, ['verb2', 'verb1']].values
    df['label'] = df['relation'].copy()
    df['relation'] = df['relation'].replace('AFTER', 'BEFORE')

    eiid1_eiid2 = list(zip(df['eiid1'], df['eiid2']))
    df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

    return df

def generate_training_dataset(gold_df: pd.DataFrame, docs_dir: Union[str, Path]) -> pd.DataFrame:
    docs = [Doc(path) for path in Path(docs_dir).glob('*.tml')]

    doc_relations = []
    for doc in docs:
        doc_relation = gold_df[gold_df['docid'] == doc.docid].copy()
        doc_relation['text'] = pd.NA

        # doc.get_text_with_relation_external()
        lines = doc.get_text().split('\n\n')

        event_dict = {}
        for idx, line in enumerate(lines):
            ids_groups = re.findall(r'(ei\d+):\w+\s*', line)
            if ids_groups:
                indexes = np.repeat(idx, len(ids_groups))
                event_dict.update(dict(zip(ids_groups, indexes)))

        plain_text_lines = re.sub(r'ei\d+:(\w+)\s*', r'\1 ', doc.get_text()).split('\n\n')

        for idx, row in doc_relation.iterrows():
            if row['eiid1'] not in event_dict:
                logging.warning(f'docid: {doc.docid} and eid: {row["eiid1"]}')
                continue

            if row['eiid2'] not in event_dict:
                logging.warning(f'docid: {doc.docid} and eid: {row["eiid2"]}')
                continue

            e1_index = event_dict[row['eiid1']]
            e2_index = event_dict[row['eiid2']]

            if e1_index != e2_index:
                min_idx = min(e1_index, e2_index)
                max_idx = max(e1_index, e2_index)

                doc_relation.at[idx, 'text'] = '\n'.join(plain_text_lines[min_idx: max_idx+1])
            else:
                doc_relation.at[idx, 'text'] = plain_text_lines[e1_index]

        # prev_idx = 0
        # for idx in range(2, len(lines) + 1):
        #     tuple_lines = lines[prev_idx:idx]
        #     text = '\n'.join(tuple_lines)
        #
        #     prev_idx = idx - 1
        #
        #     ids_groups = re.findall(r'(e\d+):\w+\s+', text)
        #     if len(ids_groups) < 2:
        #         continue
        #
        #     mask = doc_relation.eiid1.isin(ids_groups) & doc_relation.eiid2.isin(ids_groups) & doc_relation.text.isna()
        #     if not doc_relation[mask].empty:
        #         doc_relation.loc[mask, 'text'] = text

        doc_relations.append(doc_relation)

    return pd.concat(doc_relations, ignore_index=True).dropna(subset='text')

class Doc:
    def __init__(self, path: Path):
        self.path = path
        self.doc = self.parse()
        self.docid = self.doc.find('docid').get_text()
        self.mapping = self.get_mapping()
        self.text = self.get_text()
        self.relations = self.get_relations()

    def parse(self):
        with self.path.open('r') as f:
            data = f.read()
        return BeautifulSoup(data, features='lxml')

    def get_mapping(self):
        # text = self.doc.find(name="MAKEINSTANCE")
        return {d.get("eventid"): d.get("eiid") for d in self.doc.find_all('makeinstance')}

    def get_text(self):
        text = self.doc.find(name="text")
        [d.replace_with(f'{self.mapping[d.get("eid")] if d.get("eid") in self.mapping else d.get("eid")}:{d.get_text()}')  for d in text.find_all('event')]
        [d.replace_with(d.get_text()) for d in text.find_all('timex3')]
        return text.get_text().strip()

    def get_relations(self):
        relations = ((d.get('eventinstanceid'), d.get('relatedtoeventinstance'), d.get('reltype')) for d in
                     self.doc.find_all('tlink') if d.get('reltype') in set(['IBEFORE', 'BEFORE', 'AFTER', 'IAFTER']))
        return list(filter(lambda d: d[0] is not None and d[1] is not None, relations))

    def to_prompt(self) -> str:
        return f"""
    Use the uppercase letter with the prefix e[numeric] to describe the timeline graph with temporal relations between them use BEFORE and AFTER. specify only the code name aka e[numeric]. Verify yourself and answer only from the provided text ..
    For example: e1 before e2
    The text-
    {self.get_text()}
        """

if __name__ == '__main__':
    DATA_PATH = Path('../../data')
    MATRES_PATH = DATA_PATH / 'MATRES'
    DOCS_DIRS_PATH = MATRES_PATH / 'raw' / 'TBAQ-cleaned'

    BASE_NAMES = ['AQUAINT', 'TimeBank', 'te3-platinum']
    BASE_DF_PATHS = ['aquaint.txt', 'timebank.txt', 'platinum.txt']

    for name, df_name in zip(BASE_NAMES, BASE_DF_PATHS):
        gold_df = load_data(MATRES_PATH / df_name)

        df = generate_training_dataset(gold_df, docs_dir=DOCS_DIRS_PATH / name)
        output = DATA_PATH / 'trc-prepared-data'
        output.mkdir(exist_ok=True)
        df.to_csv(output / f'{name}.csv', index=False)
