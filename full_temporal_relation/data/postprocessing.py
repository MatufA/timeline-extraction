import logging
import re
import json
from pathlib import Path
from typing import List

import pandas as pd

from full_temporal_relation.data.preprocessing import Doc, load_data
from full_temporal_relation.metrics import get_n_graph_cycles


def prepare_df_from_json_response(response: List[dict], doc_obj: Doc, mode: str):
    ei_keys = set(doc_obj.mapping.values())
    data = []

    if isinstance(response, str):
        new_response = []
        for r in response.replace('[', '').replace(',', '').strip().split('\n'):
            try:
                new_response.append(json.loads(r.strip()))
            except json.JSONDecodeError as e:
                pass
    elif isinstance(response, dict):
        new_response = [response]
    else:
        new_response = response

    for relation in new_response:
        e1, e2 = (f'ei{relation["event1"]}', f'ei{relation["event2"]}') if mode != 'pair' else (relation["event1"], relation["event2"])
        relation = relation['relation']
        if e1 not in ei_keys or e2 not in ei_keys:
            unique_id = 'UNDEFINED'
        else:
            unique_id = '-'.join(sorted([e1, e2]))

        if relation == 'after':
            e1, e2 = e2, e1
            label = 'after'
            relation = 'before'
        else:
            label = relation

        data.append({
            'docid': doc_obj.docid,
            'eiid1': e1.strip(),
            'eiid2': e2.strip(),
            'relation': relation,
            'unique_id': unique_id,
            'p_label': label.upper(),
            'mode': mode
        })
    return pd.DataFrame(data)


def prepare_df_from_response(response: str, doc_obj: Doc, mode: str):
    supported_relations = ('before', 'after', 'equal', 'vague')
    ei_keys = set(doc_obj.mapping.values())
    data = []
    for line in response.strip().splitlines():
        e1e2_opt = re.search(r'e.*', line)
        if e1e2_opt is None:
            continue

        e1e2 = e1e2_opt.group(0).strip()

        regex_parsers = [r'(ei\d+):\w+\s+(\w+)\s+(ei\d+).*',
                         r'(ei\d+)\s+(\w+)\s+(ei\d+).*',
                         r'(ei\d+)\s+\w+\s+(\w+)\s+(ei\d+).*',
                         r'(ei\d+):(\w+)\s+(ei\d+).*',
                         r'(ei\[\d+\])\s+(\w+)\s+(ei\[\d+\]).*',
                         r'(ei\d+)\s+\[\w+\]\s+(ei\d+):\s+(\w+)',
                         r'(ei\d+)\s+\[(\w+)\]\s+(ei\d+)',
                         r'(ei\d+):\s*\w+\s+(\w+)\s+(ei\d+).*',
                         ]

        e1_relation_e2 = None
        for regex_parser in regex_parsers:
            e1_relation_e2 = re.search(regex_parser, e1e2.lower())
            if e1_relation_e2 is not None:
                break

        if e1_relation_e2 is None:
            logging.error(f'unable to parse line: "{line}", ignored line')
            continue

        e1, relation, e2 = e1_relation_e2.groups()
        e1 = e1.strip()
        e2 = e2.strip()
        relation = relation.strip()

        if relation not in supported_relations:
            if e2 in supported_relations:
                e2, relation= relation, e2
            else:
                continue

        if e1 not in ei_keys or e2 not in ei_keys:
            unique_id = 'UNDEFINED'
            # verb1 = doc_obj.mapping.get(e1)
            # verb2 = doc_obj.mapping.get(e2)
        else:
            unique_id = '-'.join(sorted([e1, e2]))
            # verb1 = doc_obj.mapping[e1]
            # verb2 = doc_obj.mapping[e2]

        if relation == 'after':
            # verb1, verb2 = verb2, verb1
            e1, e2 = e2, e1
            label = 'after'
            relation = 'before'
        else:
            label = relation

        data.append({
            'docid': doc_obj.docid,
            # 'verb1': verb1,
            # 'verb2': verb2,
            'eiid1': e1.strip(),
            'eiid2': e2.strip(),
            'relation': relation,
            'unique_id': unique_id,
            'p_label': label.upper(),
            'mode': mode
        })
    return pd.DataFrame(data)


def transform_to_before(df, doc_id):
    doc_events = df[df.docid == doc_id].copy()
    doc_events = doc_events.apply(
        lambda row: (row.verb2, row.verb1, row.eiid2, row.eiid1, 'BEFORE') if row.relation == 'AFTER' else (
            row.verb1, row.verb2, row.eiid1, row.eiid2, row.relation), axis='columns', result_type="expand")
    return doc_events


def majority_vote_decision(parsed_response_path: Path, results_path: Path, min_votes: int = 3):
    predicted_df = pd.read_csv(parsed_response_path)

    predicted_df['score'] = 1
    predicted_sum_df = predicted_df.groupby(['docid', 'unique_id', 'relation'])['score'].sum().reset_index()
    predicted_sum_df = predicted_sum_df[predicted_sum_df['score'] >= min_votes]

    agg_preds = []
    for (docid, unique_id), group in predicted_sum_df.groupby(['docid', 'unique_id']):
        max_score = group.score.max()
        relations = group[group.score == max_score]['relation'].to_list()

        if len(relations) > 1:
            agg_preds.append({
                'docid': docid,
                'unique_id': unique_id,
                'relation': 'VAGUE',
                'max_score': max_score,
                'conflict_rel': ','.join(relations)
            })
        else:
            agg_preds.append({
                'docid': docid,
                'unique_id': unique_id,
                'relation': relations[0].upper(),
                'max_score': max_score
            })

    agg_pred_df = pd.DataFrame(agg_preds)
    predicted_df.relation = predicted_df.relation.str.upper()
    merge_cols = ['docid', 'unique_id', 'relation']
    predicted_agg_df = (pd.merge(predicted_df, agg_pred_df, on=merge_cols, how='inner')
                        .drop_duplicates(merge_cols))

    cycles = get_n_graph_cycles(predicted_agg_df)

    cycles_df = pd.DataFrame.from_dict(cycles, orient='index', columns=['ccr']).reset_index()
    cycles_df.columns = ['docid', 'n_cycles']

    final_df = pd.merge(predicted_df, agg_pred_df,
                        on=['docid', 'unique_id'],
                        suffixes=(None, "_selected"),
                        how='left')
    final_df = pd.merge(final_df, cycles_df, on=['docid'], how='left')
    final_df['min_vote'] = min_votes

    final_df.to_csv(results_path, index=False)
