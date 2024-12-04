from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

from full_temporal_relation.data.preprocessing import load_data
from full_temporal_relation.graph import Graph


def summary_results(model_results_path: Path, gold_df: pd.DataFrame, model_name: str, target_col:str='label'):
    relations = ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']
    df__results = pd.read_csv(model_results_path)
    preds_df = (
        df__results[['docid', 'unique_id', 'relation_selected', 'p_label']]
        .copy()
        .dropna(axis='rows')
        .drop_duplicates(['docid', 'unique_id', 'relation_selected'])
        .rename({'relation_selected': 'relation'}, axis='columns'))
    
    # non_vague_gold_df = gold_df[gold_df['label'] != 'VAGUE']
    # joined_df = pd.merge(preds_df, non_vague_gold_df[['docid', 'unique_id', 'label']], how='inner',
    #                      on=['docid', 'unique_id',])
    # precision, recall, f1, support = precision_recall_fscore_support(joined_df.label, joined_df.p_label, average='samples', labels=joined_df.p_label.unique())

    # grouped results by relation type, results summarization
    relation_grouped_df = relation_table(gold_df, preds_df, model_name, target_col=target_col)

    # calculate cycles
    matched_preds_df = pd.merge(preds_df, gold_df[['docid', 'unique_id']], how='inner',
                             on=['docid', 'unique_id'])
    coverage_score = matched_preds_df.shape[0]/gold_df.shape[0]
    cycles_score = (f"{df__results[['docid', 'n_cycles']].drop_duplicates()['n_cycles'].sum()} / "
                    f"{df__results['docid'].nunique()}")

    precision_df = precision(relation_grouped_df).fillna(0)
    recall_df = recall(relation_grouped_df).fillna(0)
    f1_df = calculate_f1(precision_df.to_numpy()[0], recall_df.to_numpy()[0])
    micro_f1_score = calculate_micro_f1(precision_df.to_numpy()[0], recall_df.to_numpy()[0])
    relax_micro_f1 = calculate_relax_micro_f1(precision_df.to_numpy()[0], recall_df.to_numpy()[0])

    data_vecs = np.array(list(zip(precision_df.values.tolist()[0], recall_df.values.tolist()[0], f1_df.tolist())))
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([relations, ['precision', 'recall', 'f1']],
                                                         names=['relation', 'metric']),
                      data=data_vecs.flatten()[np.newaxis, :])
    df['micro-f1'] = micro_f1_score
    df['relax-micro-f1'] = relax_micro_f1
    df['cycles'] = cycles_score
    df['coverage'] = coverage_score
    return df


def relation_table(gold_df, preds_df, model_name, target_col='relation'):
    relevant_preds_df = pd.merge(preds_df, gold_df[['docid', 'unique_id']], how='inner',
                                 on=['docid', 'unique_id'])#.replace({'AFTER': 'BEFORE'})

    outer_no_preds_merged = pd.merge(gold_df, preds_df[['docid', 'unique_id']],
                                     how='outer',
                                     on=['docid', 'unique_id'],
                                     indicator=True)
    no_preds_df = outer_no_preds_merged[outer_no_preds_merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    # no_preds_df = no_preds_df.replace({'AFTER': 'BEFORE'}).relation.value_counts()
    no_preds_df = no_preds_df[target_col].value_counts()

    outer_no_label_merged = pd.merge(preds_df, gold_df[['docid', 'unique_id', 'label']],
                                     how='outer',
                                     on=['docid', 'unique_id'],
                                     indicator=True)
    no_label_df = outer_no_label_merged[outer_no_label_merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    # no_label_df = no_label_df.replace({'AFTER': 'BEFORE'}).relation.value_counts()
    no_label_df = no_label_df.p_label.value_counts()

    relation = ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']
    results = pd.DataFrame(index=relation, columns=relation + ['no_predictions'])
    for relation, group in gold_df.groupby(target_col):
        relevant_gold_df = pd.merge(relevant_preds_df, group[['docid', 'unique_id', target_col]], how='inner',
                                    on=['docid', 'unique_id'])
        group_value_counts = relevant_gold_df.p_label.value_counts()
        group_value_counts['no_predictions'] = no_preds_df[relation] if relation in no_preds_df else 0
        results.loc[relation] = group_value_counts

    results.loc['no_label'] = no_label_df
    results = results.infer_objects(copy=False).fillna(0)

    results['sum'] = results.sum(axis=1)
    results.loc['sum'] = results.sum(axis=0)
    results.loc['sum', 'sum'] = np.nan

    results.index = pd.MultiIndex.from_product([['gold-labeled'], results.index])
    results.columns = pd.MultiIndex.from_product([[model_name], results.columns])

    return results


def recall(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.loc[df.index[:4].values, :]
    sum_per_label = df.loc[df.index[:4].values, df.columns[-1]].values
    labels_values = np.diag(labels)
    return pd.DataFrame(columns=[col[1] for col in df.columns][:4], data=[labels_values / sum_per_label])


def precision(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.loc[df.index[:4].values, :]
    sum_per_label = df.loc[df.index[-1], df.columns[:4].values].values
    labels_values = np.diag(labels)
    return pd.DataFrame(columns=[col[1] for col in df.columns][:4], data=[labels_values / sum_per_label])


def calculate_f1(precision, recall):
    """Calculate the F1 score from precision and recall."""
    return np.array([2 * (p * r) / (p + r) if p + r != 0 else 0. for p, r in zip(precision, recall)])


def calculate_micro_f1(precisions, recalls):
    """Calculate the micro F1 score from a list of precisions and recalls."""
    total_tp = sum(p * r / (2 * p - r) for p, r in zip(precisions, recalls) if p + r != 0)
    total_fp_fn = sum(1 / (2 * p - r) for p, r in zip(precisions, recalls) if p + r != 0)

    if total_fp_fn == 0:
        return 0.0
    return total_tp / total_fp_fn

def calculate_relax_micro_f1(precisions, recalls):
    """Calculate the relax (w/o VAGUE) micro F1 score from a list of precisions and recalls."""
    return calculate_micro_f1(precisions[:-1], recalls[:-1])


def simple_consistency_rate(df: pd.DataFrame) -> dict:
    """
    Calculates the consistency rate (SCR), which count the number of event pairs in the
    test set whose reversed pair has the consistent prediction with the original pair

    :param df: the test predicted pairs
    :type df: pd.DataFrame
    :return: SCR value
    :rtype: float
    """
    results = {}

    for doc_id, group in df.groupby('docid'):

        all_pairs = group.shape[0]
        pairs_sets = set()
        consistent_pair = all_pairs

        for _, row in group.iterrows():
            if row['relation'].upper() == 'BEFORE':
                pair = (row['eiid2'], row['eiid1'])  # swap for checking if After exists
                consistent_pair = consistent_pair - 1 if pair in pairs_sets else consistent_pair
                pairs_sets.add(pair)
            elif row['relation'].upper() == 'AFTER':
                pair = (row['eiid1'], row['eiid2'])  # swap for checking if Before exists
                consistent_pair = consistent_pair - 1 if pair in pairs_sets else consistent_pair
                pairs_sets.add(pair)
            else:
                pairs = {(row['eiid1'], row['eiid2']), (row['eiid2'], row['eiid1'])}
                consistent_pair = consistent_pair - 1 if len(pairs_sets.intersection(pairs)) > 0 else consistent_pair
                pairs_sets.update(pairs)

        results[doc_id] = consistent_pair / all_pairs

    return results


def correct_consistency_rate(df_predicted: pd.DataFrame, df_true: pd.DataFrame) -> dict:
    results = {}
    scr = simple_consistency_rate(df_predicted)

    for doc_id, predicted_df in df_predicted.groupby('docid'):
        true_df = df_true.loc[df_true['docid'] == doc_id, :]

        all_pairs = true_df.shape[0]
        correct_event_count = predicted_df.merge(true_df, on=['unique_id', 'relation']).dropna().shape[0]

        results[doc_id] = correct_event_count / all_pairs  #+ scr[doc_id]

    return results


def get_n_graph_cycles(df: pd.DataFrame) -> dict:
    return {k: len(v) for k, v in Graph().find_cycles(df).items()}


if __name__ == '__main__':
    # model_name = 'llama-3.1-70b-versatile'
    # model_name = 'gemini-1.5-pro'
    # model_name = 'gemini-1.5-flash'
    method = 'few-shot'  #  'zero-shot'
    mode = 'multi'  # 'pair'
    data_name = 'te3-platinum'
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    suffixes = [model_name.split('/')[1] if '/' in model_name else model_name, method]
    suffix_name = '-'.join(suffixes)

    MATRES_DATA_PATH = Path('../data')
    TRC_RAW_PATH = MATRES_DATA_PATH / 'TRC'
    # parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'platinum-test-results-{model_name}.csv'
    parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'{mode}-{data_name}-results-{suffix_name}.csv'
    gold_data_path = MATRES_DATA_PATH / 'MATRES' / 'platinum.txt'
    results_path = TRC_RAW_PATH / 'results' / method / f'platinum-results-{model_name}.csv'
    min_vote = 3

    predicted_df = pd.read_csv(parsed_response_path)
    platinum_df = load_data(gold_data_path)

    # platinum_df = pd.read_csv(gold_data_path, sep='\t', header=None,
    #                           names=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2', 'relation'])
    # platinum_df.eiid1 = 'ei' + platinum_df.eiid1.astype(str)
    # platinum_df.eiid2 = 'ei' + platinum_df.eiid2.astype(str)
    #
    # eiid1_eiid2 = list(zip(platinum_df['eiid1'], platinum_df['eiid2']))
    # platinum_df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

    predicted_df['score'] = 1
    predicted_sum_df = predicted_df.groupby(['docid', 'unique_id', 'relation'])['score'].sum().reset_index()
    predicted_sum_df = predicted_sum_df[predicted_sum_df['score'] > min_vote]

    agg_preds = []
    for (docid, unique_id), group in predicted_sum_df.groupby(['docid', 'unique_id']):
        max_score = group.score.max()
        relations = group[group.score == max_score]['relation'].to_list()
        group_df = group.reset_index()

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

    scr = simple_consistency_rate(predicted_agg_df)
    ccr = correct_consistency_rate(df_predicted=predicted_agg_df.drop('conflict_rel', axis=1, errors='ignore'),
                                   df_true=platinum_df)
    cycles = get_n_graph_cycles(predicted_agg_df)

    scr_df = pd.DataFrame.from_dict(scr, orient='index', columns=['scr']).reset_index()
    scr_df.columns = ['docid', 'scr']

    ccr_df = pd.DataFrame.from_dict(ccr, orient='index', columns=['ccr']).reset_index()
    ccr_df.columns = ['docid', 'ccr']

    cycles_df = pd.DataFrame.from_dict(cycles, orient='index', columns=['ccr']).reset_index()
    cycles_df.columns = ['docid', 'n_cycles']

    metrics_df = pd.merge(ccr_df, scr_df, on='docid', how='inner')
    metrics_df = pd.merge(metrics_df, cycles_df, on='docid', how='left')

    final_df = pd.merge(predicted_df, agg_pred_df,
                        on=['docid', 'unique_id'],
                        suffixes=(None, "_selected"),
                        how='left')
    final_df = pd.merge(final_df, metrics_df, on=['docid'], how='left')
    final_df['min_vote'] = min_vote

    final_df.to_csv(results_path, index=False)

    print(f'[{model_name}] metric results: SCR: {scr} and CCR: {ccr}')
