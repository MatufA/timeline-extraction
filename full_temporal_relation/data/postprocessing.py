import logging
import re
import pandas as pd

from full_temporal_relation.data.preprocessing import Doc


def prepare_df_from_response(response: str, doc_obj: Doc):
    data = []
    for line in response.strip().splitlines():
        e1e2_opt = re.search(r'e.*', line)
        if e1e2_opt is None:
            continue
        e1e2 = e1e2_opt.group(0)
        e1e2_lower = e1e2.strip().lower()
        if 'before' in e1e2_lower:
            relation = 'before'
        elif 'after' in e1e2_lower:
            relation = 'after'
        elif 'equal' in e1e2_lower:
            relation = 'equal'
        elif 'vague' in e1e2_lower:
            relation = 'vague'
        else:
            continue

        split_e1_e2 = e1e2.split(f' {relation} ')
        if len(split_e1_e2) == 1:
            split_e1_e2 = e1e2.split(f' {relation.upper()} ')

        if len(split_e1_e2) != 2:
            logging.error(f'unable to parse line: "{line}", ignored line')
            continue
        e1, e2 = split_e1_e2

        if e1 not in doc_obj.mapping or e2 not in doc_obj.mapping:
            unique_id = 'UNDEFINED'
            verb1 = doc_obj.mapping.get(e1)
            verb2 = doc_obj.mapping.get(e2)
        else:
            unique_id = '-'.join(sorted([e1, e2]))
            verb1 = doc_obj.mapping[e1]
            verb2 = doc_obj.mapping[e2]

        if relation == 'after':
            verb1, verb2 = verb2, verb1
            e1, e2 = e2, e1
            relation = 'before'

        data.append({
            'docid': doc_obj.docid,
            'verb1': verb1,
            'verb2': verb2,
            'eiid1': e1.strip(),
            'eiid2': e2.strip(),
            'relation': relation,
            'unique_id': unique_id
        })
    return pd.DataFrame(data)


def transform_to_before(df, doc_id):
    doc_events = df[df.docid == doc_id].copy()
    doc_events = doc_events.apply(
        lambda row: (row.verb2, row.verb1, row.eiid2, row.eiid1, 'BEFORE') if row.relation == 'AFTER' else (
            row.verb1, row.verb2, row.eiid1, row.eiid2, row.relation), axis='columns', result_type="expand")
    return doc_events
