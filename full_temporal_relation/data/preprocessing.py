import re
import json
import logging
from typing import Union, Literal

import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

def replace_eid(text, exclude_ids):
    pattern = r'ei(\d+):(\w+)'
    
    def replace_func(match):
        ei_id = match.group(1)
        word = match.group(2)
        
        if f'ei{ei_id}' in exclude_ids:
            return f'[EVENT{ei_id}]{word}[/EVENT{ei_id}]'
        else:
            return word
    
    # Use re.sub with a replacement function
    result = re.sub(pattern, replace_func, text)
    return result


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None,
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

def generate_training_dataset(gold_df: pd.DataFrame, docs_dir: Union[str, Path], mode: str) -> pd.DataFrame:
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

        # plain_text_lines = re.sub(r'ei\d+:(\w+)\s*', r'\1 ', doc.get_text()).split('\n\n')
        plain_text_lines = doc.get_text().split('\n\n')

        for idx, row in doc_relation.iterrows():
            if row['eiid1'] not in event_dict:
                logging.warning(f'docid: {doc.docid} and eid: {row["eiid1"]}')
                continue

            if row['eiid2'] not in event_dict:
                logging.warning(f'docid: {doc.docid} and eid: {row["eiid2"]}')
                continue

            e1_index = event_dict[row['eiid1']]
            e2_index = event_dict[row['eiid2']]

            if mode == 'pair':
                if e1_index != e2_index:
                    min_idx = min(e1_index, e2_index)
                    max_idx = max(e1_index, e2_index)

                    doc_relation.at[idx, 'text'] = '\n'.join(plain_text_lines[min_idx: max_idx+1])
                else:
                    doc_relation.at[idx, 'text'] = plain_text_lines[e1_index]
            elif mode == 'multi':
                doc_relation.at[idx, 'text'] = '\n'.join(plain_text_lines)

        doc_relations.append(doc_relation)

    return pd.concat(doc_relations, ignore_index=True).dropna(subset='text')

def prepare_as_jsonl(df: pd.DataFrame, output_path: Path, mode: Literal['multi', 'pair']) -> None:

    data = []

    if mode == 'pair':
        for idx, row in df.iterrows():
            data.append({
                'text': replace_eid(row.text, exclude_ids=[row.eiid1, row.eiid2]),
                'relations': [(row.eiid1, row.eiid2)],
                'doc_id': row.docid, 
                'true_label': row.label.lower()
            })
    else:
        for doc_id, group in df.groupby('docid'):
            # relations = '\n'.join(group.loc[:, ['eiid1', 'eiid2']].apply(lambda row: ' [RELATION] '.join(row), axis=1))
            unique_relation_ids = np.unique(group.loc[:, ['eiid1', 'eiid2']].to_numpy().flatten())
            data.append({
                'text': replace_eid(group.iloc[0].text, exclude_ids=unique_relation_ids),
                'relations': group.loc[:, ['eiid1', 'eiid2']].apply(lambda row: tuple(row), axis=1).values.tolist(),
                'doc_id': doc_id, 
                'true_label': group.loc[:, 'label'].to_numpy().tolist()
            })

    output_path.write_text(json.dumps(data, indent=2))


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
        eid_mapping = {}
        for d in self.doc.find_all('makeinstance'):
            if d.get("eventid") not in eid_mapping:
                eid_mapping[d.get("eventid")] = d.get("eiid")
        return eid_mapping

    def get_text(self):
        text = self.doc.find(name="text")
        [d.replace_with(f'{self.mapping[d.get("eid")] if d.get("eid") in self.mapping else d.get("eid")}:{d.get_text()}')  for d in text.find_all('event')]
        [d.replace_with(d.get_text()) for d in text.find_all('timex3')]
        return text.get_text().strip()

    def get_relations(self):
        relations = ((d.get('eventinstanceid'), d.get('relatedtoeventinstance'), d.get('reltype')) for d in
                     self.doc.find_all('tlink') if d.get('reltype') in set(['IBEFORE', 'BEFORE', 'AFTER', 'IAFTER']))
        return list(filter(lambda d: d[0] is not None and d[1] is not None, relations))


if __name__ == '__main__':
    DATA_PATH = Path('./data')
    MATRES_PATH = DATA_PATH / 'MATRES'
    DOCS_DIRS_PATH = MATRES_PATH / 'raw' / 'TBAQ-cleaned'

    BASE_NAMES = ['AQUAINT', 'TimeBank', 'te3-platinum']
    BASE_DF_PATHS = ['aquaint.txt', 'timebank.txt', 'platinum.txt']
    modes = ['pair', 'multi']

    for mode in modes: 

        if mode == 'pair':
            output = DATA_PATH / 'wo-vague' / 'trc-prepared-data'
        elif mode == 'multi':
            output = DATA_PATH / 'wo-vague' / 'te-prepared-data'

        for name, df_name in zip(BASE_NAMES, BASE_DF_PATHS):
            output_file = output / f'{mode}-{name}.csv'

            if not output_file.exists():
                gold_df = load_data(MATRES_PATH / df_name)
                gold_df = gold_df[gold_df.label != 'VAGUE']
                df = generate_training_dataset(gold_df, docs_dir=DOCS_DIRS_PATH / name, mode=mode)
                output.mkdir(exist_ok=True, parents=True)
                df.to_csv(output_file, index=False)
            else:
                df = pd.read_csv(output_file, header=0)
                print(f"Data already exists for {name}. Skipping.")

            prepare_as_jsonl(df,
                            output_path=DATA_PATH/'TRC'/'raw_text'/f'{mode}_{name.lower()}_text_w_relations_prepared.json',
                            mode=mode)
