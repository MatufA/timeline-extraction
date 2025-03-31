from collections import Counter
import pandas as pd
from pathlib import Path
import json

from tqdm.auto import tqdm

from full_temporal_relation.data.preprocessing import Doc, load_data, replace_eid
from full_temporal_relation.models.OpenAIClient import OpenAIClient, TemporalRelationsToDrop
from full_temporal_relation.models.LLModel import JsonParser, NoParser
from full_temporal_relation.prompts.Prompt import BreakCycleEvents, BreakCycleEventsByRelabeling, NTBreakCycleEvents
from full_temporal_relation.pipeline import get_summary_results
from full_temporal_relation.metrics import summary_results
from full_temporal_relation.graph import Graph

def prepare_annotation(df, output_doc: Path, docs):
    with output_doc.open('w') as j_file:
            for docid, group in df.groupby('docid'):
                if 'sarcozy' in docid:
                    docid = 'nyt_20130321_sarkozy'
                doc = docs[docid]
                eiids = set(group.eiid1.values.tolist() + group.eiid2.values.tolist())
                excluded = [eiid.replace('e', 'ei') if 'ei' not in eiid else eiid for eiid in eiids]
                annotated_text = replace_eid(doc.get_text(), exclude_ids=excluded)
                line = json.dumps({
                    'docid': docid,
                    'text': annotated_text, 
                    'relations': group[['eiid1', 'eiid2', 'probs', 'relation', 'unique_id']].to_dict(orient='records')
                })
                j_file.write(line + '\n')

def prepare_cycles_only_df(cycles, df):
    data = []
    for docid, cycles in cycles.items():
        for cycle in cycles:
            data.extend([docid, eiid1.split('-')[1], eiid2.split('-')[1]] for eiid1, eiid2, relation in cycle)
    
    df_cycles = pd.DataFrame(data, columns=['docid', 'eiid1', 'eiid2'])
    eiid1_eiid2 = list(zip(df_cycles['eiid1'], df_cycles['eiid2']))
    df_cycles['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

    cycle_only_df = pd.merge(df, df_cycles[['docid', 'unique_id']], on=['docid', 'unique_id'], how='inner').drop_duplicates()
    return cycle_only_df

def prepare_narrativetime_annotation_recoreds(df, docs):
    records = []
    for docid, group in df.groupby('docid'):
        full_text = docs[docid]
        eiids = set(group.eiid1.values.tolist() + group.eiid2.values.tolist())
        annotated_text = replace_eid(full_text, exclude_ids=[eiid.replace('e', 'ei') for eiid in eiids])
        records.append({
            'docid': docid,
            'text': annotated_text, 
            'relations': group[['eiid1', 'eiid2', 'probs', 'relation', 'unique_id']].to_dict(orient='records')
        })
    return records

def prepare_annotation_recoreds(df, docs):
    records = []
    for docid, group in df.groupby('docid'):
        if 'sarcozy' in docid:
            doc = docs['nyt_20130321_sarkozy']
        else:
            doc = docs[docid]
        eiids = set(group.eiid1.values.tolist() + group.eiid2.values.tolist())
        annotated_text = replace_eid(doc.get_text(), exclude_ids=[eiid.replace('e', 'ei') for eiid in eiids])
        group['eiid1'] = group['eiid1'].str.replace('e', 'EVENT')
        group['eiid2'] = group['eiid2'].str.replace('e', 'EVENT')
        records.append({
            'docid': docid,
            'text': annotated_text, 
            'relations': group[['eiid1', 'eiid2', 'probs', 'relation', 'unique_id']].to_dict(orient='records')
        })
    return records

def evaluate_by_relabel(model_name, prompt_template, data_path, cycley_data, result_path_name):
    results_path = data_path / result_path_name.format(model_name=model_name)
    model = OpenAIClient(model_name=model_name, use_formate=False, parser=JsonParser, use_dot_graph=True)
    min_votes = 3

    with results_path.open('w') as file:
            for record in tqdm(cycley_data, desc='Text evaluation', position=0, leave=True):
                prompt = prompt_template.generate_dict_prompt(**{p: record[p] for p in ['text', 'relations']})
                response = model.generate_response(prompt)

                model_response = []
                for res in model.prepare_response(response):
                    res['docid'] = record['docid']
                    if not res['response']:
                        print(f'unable to process res: {response}')
                        continue

                    for r in res['response']:
                        eiid1 = r['event1'].replace('EVENT', 'ei')
                        eiid2 = r['event2'].replace('EVENT', 'ei')
                        unique_id = '-'.join(sorted([eiid1, eiid2]))
                        model_response.append({'eiid1': eiid1, 
                                               'eiid2': eiid2, 
                                               'unique_id': unique_id, 
                                               'docid': record['docid'], 
                                               'p_label': r['relation'], 
                                               'relation': r['relation']})

                df_response = pd.DataFrame(model_response)

                # Group by docid, unique_id, and label, count votes
                vote_counts = df_response.groupby(['docid', 'unique_id', 'p_label']).size().reset_index(name='vote_count')
                
                # Find rows with at least min_votes
                qualified_votes = vote_counts[vote_counts['vote_count'] >= min_votes]
                
                # For each (docid, unique_id), select the label with max votes
                majority_labels = qualified_votes.loc[qualified_votes.groupby(['docid', 'unique_id'])['vote_count'].idxmax()]
                
                # Merge back with original dataframe to get full row details
                filtered_df = pd.merge(df_response.drop_duplicates(subset=['docid', 'unique_id', 'p_label']), 
                                    majority_labels[['docid', 'unique_id', 'p_label']], 
                                    on=['docid', 'unique_id', 'p_label'], 
                                    how='inner')
                
                for rec in filtered_df.to_dict(orient='records'):
                    json_line = json.dumps(rec)
                    file.write(json_line + '\n')

                

def evaluate(model_name, prompt_template, data_path, cycley_data, result_path_name):
    results_path = data_path / result_path_name.format(model_name=model_name)
    model = OpenAIClient(model_name=model_name, use_formate=False, parser=NoParser, use_dot_graph=False, response_format=TemporalRelationsToDrop)
    # records = json.load(cycle_data_path.open('r'))

    with results_path.open('w') as file:
        for record in tqdm(cycley_data, desc='Text evaluation', position=0, leave=True):
            prompt = prompt_template.generate_dict_prompt(**{p: record[p] for p in ['text', 'relations']})
            response = model.generate_response(prompt)

            model_response = [{
                'trail': idx,
                'response': choice.message.parsed.unique_ids ,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            } for idx, choice in enumerate(response.choices)]

            count = Counter([idx for res in model_response for idx in res['response']])
            max_choosen = max(count.values())
            majority_vote_drop_ids = [index for index, freq in count.items() if freq == max_choosen]

            df_relations = pd.DataFrame(record['relations'])
            drop_edges = df_relations.iloc[majority_vote_drop_ids, :].copy()
            # df_relations = df_relations[~df_relations.unique_id.isin(drop_edges.unique_id)]
            drop_edges.loc[:, 'docid'] = record['docid']
            
            for rec in drop_edges.to_dict(orient='records'):
                json_line = json.dumps(rec)
                file.write(json_line + '\n')
    


if __name__ == '__main__':
    DATA_PATH = Path('./data')
    MATRES_PATH = DATA_PATH / 'MATRES'
    DOCS_DIRS_PATH = MATRES_PATH / 'raw' / 'TBAQ-cleaned'
    preparation_cycles = DATA_PATH / 'te3-platinum_cycles_only.json'
    result_path_name = '{model_name}_narrativetime_a1_predictions_cycles_only_results.csv'
    gold_data_path = MATRES_PATH / 'platinum.txt'

    TRC_GLOBAL_VARIABLES = {
    "LABELS": ["BEFORE", "AFTER", "EQUAL", "VAGUE"],
    "LABELS_NO_VAGUE": ["BEFORE", "AFTER", "EQUAL"],
    "LABELS_IDS": [0, 1, 2, 3]
    }
    id2label = dict(zip(TRC_GLOBAL_VARIABLES['LABELS_IDS'], TRC_GLOBAL_VARIABLES['LABELS']))

    platinum_df = load_data(gold_data_path)
    # nt_a1_test_gold = pd.read_csv(DATA_PATH / 'narrativetime_a1_test_gold.csv')

    initial_preds_df = pd.read_csv(DATA_PATH / 'grouped_cycles_matres_by_baseline.csv')
    # predicted_df = pd.read_csv(DATA_PATH / 'comb_te3-platinum_minimal_context_predictions_with_probs_baseline.csv')
    # eiid1_eiid2 = list(zip(predicted_df['eiid1'], predicted_df['eiid2']))
    # predicted_df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

    # initial_preds_df = pd.merge(initial_preds_df, predicted_df[['docid', 'unique_id', 'predictions', 'probs']], on=['docid', 'unique_id'], how='left')

    initial_preds_df = initial_preds_df.rename(columns={'doc_id': 'docid'})
    initial_preds_df['relation'] = initial_preds_df.predictions.replace(id2label)
    eiid1_eiid2 = list(zip(initial_preds_df['eiid1'], initial_preds_df['eiid2']))
    initial_preds_df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

    # cycles_df = pd.read_csv(DATA_PATH / 'comb_te3-platinum_minimal_context_predictions_cycles_only.csv')
    docs = {path.name[:-4]:Doc(path) for path in Path(DOCS_DIRS_PATH / 'te3-platinum').glob('*.tml')}
    # docs = json.load((DATA_PATH / 'narrativetime_a1_test_to_text_mappings.json').open('r'))

    # prepare_annotation(cycles_df, preparation_cycles, docs)

    model_names = ['gpt-4o-mini', 'gpt-4o'] 
    # prompt_template = NTBreakCycleEvents()
    # prompt_template = BreakCycleEvents()
    prompt_template = BreakCycleEventsByRelabeling()
    results = []
    graph = Graph(use_equal=False, supported_relation=['AFTER', 'BEFORE'])
    edges_to_drop = []

    for model_name in model_names:
        predicted_df = initial_preds_df
        while sum(len(v) for v in graph.find_cycles(predicted_df).values()) > 0:
            cycles = {k: v for k, v in graph.find_cycles(predicted_df).items()}
            n_cycles = sum(len(v) for v in cycles.values())
            print(f'#cycles={n_cycles}')
            for cycle_id, cycles_only_df in predicted_df.groupby('cycle_id'):
                # cycles_only_df = prepare_cycles_only_df(cycles, predicted_df)
                recoreds = prepare_annotation_recoreds(cycles_only_df, docs)
                # recoreds = prepare_narrativetime_annotation_recoreds(cycles_only_df, docs)

                # evaluate(model_name, prompt_template, 
                #         data_path=DATA_PATH, 
                #         cycley_data=recoreds, 
                #         result_path_name=result_path_name)

                evaluate_by_relabel(model_name, prompt_template, 
                        data_path=DATA_PATH, 
                        cycley_data=recoreds, 
                        result_path_name=result_path_name)
                
                predicted_edges_to_drop_df = pd.read_json(DATA_PATH / result_path_name.format(model_name=model_name), lines=True)
                edges_to_drop.append(predicted_edges_to_drop_df)
                
                df_filtered = pd.merge(predicted_df, predicted_edges_to_drop_df[['docid', 'unique_id']], on=['docid', 'unique_id'], how='left', indicator=True)
                df_no_cycles = df_filtered[df_filtered['_merge'] != 'both'].drop('_merge', axis=1)

                predicted_df = df_no_cycles
                predicted_df['p_label'] = predicted_df['relation'] 

        # predicted_df.eiid1 = predicted_df.eiid1.str.replace('e', 'ei')
        # predicted_df.eiid2 = predicted_df.eiid2.str.replace('e', 'ei')
        # predicted_df.unique_id = predicted_df.unique_id.str.replace('e', 'ei')
        predicted_df.to_csv(DATA_PATH / f'{model_name}_narrativetime_predictions_no_cycles_by_llm.csv', index=False)
        results_df = summary_results(model_results_df=predicted_df, gold_df=platinum_df, model_name=model_name)
        results_df.to_csv(DATA_PATH / f'results_narrativetime_a1_test_no_cycles_by_{model_name}.csv', index=False)
        results.append(results_df)
        
        print('Done')

    

