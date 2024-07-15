from pathlib import Path

import pandas as pd

from full_temporal_relation.graph import Graph


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

        results[doc_id] = correct_event_count / all_pairs #+ scr[doc_id]

    return results


def get_n_graph_cycles(df: pd.DataFrame) -> dict:
    return {k: len(v) for k, v in Graph().find_cycles(df).items()}


if __name__ == '__main__':
    # model_name = 'gemini-1.5-pro'
    model_name = 'gemini-1.5-flash'
    method = 'zero-shot'

    MATRES_DATA_PATH = Path('../data')
    TRC_RAW_PATH = MATRES_DATA_PATH / 'TRC'
    parsed_response_path = TRC_RAW_PATH / 'parsed_responses' / method / f'platinum-test-results-{model_name}.csv'
    gold_data_path = MATRES_DATA_PATH / 'MATRES' / 'platinum.txt'
    results_path = TRC_RAW_PATH / 'results' / method / f'platinum-results-{model_name}.csv'
    min_vote = 0

    predicted_df = pd.read_csv(parsed_response_path)

    platinum_df = pd.read_csv(gold_data_path, sep='\t', header=None,
                              names=['docid', 'verb1', 'verb2', 'eiid1', 'eiid2', 'relation'])
    platinum_df.eiid1 = 'e' + platinum_df.eiid1.astype(str)
    platinum_df.eiid2 = 'e' + platinum_df.eiid2.astype(str)

    eiid1_eiid2 = list(zip(platinum_df['eiid1'], platinum_df['eiid2']))
    platinum_df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]

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
    ccr = correct_consistency_rate(df_predicted=predicted_agg_df.drop('conflict_rel', axis=1), df_true=platinum_df)
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