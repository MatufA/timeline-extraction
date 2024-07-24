from typing import Union

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup


def load_data(paht: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(paht, sep='\t', header=None,
                     names=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2', 'relation'])
    df.eiid1 = 'e' + df.eiid1.astype(str)
    df.eiid2 = 'e' + df.eiid2.astype(str)

    mask = df['relation'] == 'after'
    df.loc[mask, ['verb1', 'verb2']] = df.loc[mask, ['verb2', 'verb1']].values
    df['relation'] = df['relation'].replace('AFTER', 'BEFORE')

    return df


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
        text = self.doc.find(name="text")
        return {d.get("eid"): d.get_text() for d in text.find_all('event') if d.get("eid") is not None}

    def get_text(self):
        text = self.doc.find(name="text")
        [d.replace_with(f'{d.get("eid")}:{d.get_text()}') for d in text.find_all('event')]
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
