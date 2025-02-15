# %%
from pathlib import Path
import pandas as pd

DATA_PATH = Path('../../data')
TRC_RAW_PATH = DATA_PATH / 'TRC'

relations_dfs = []
for model in ['gpt-4o-mini', 'gpt-4o']:
    for method in ['zero-shot', 'few-shot']:
        results_path = TRC_RAW_PATH / 'results' / method / f'multi-te3-platinum-results-{model}-{method}-completion.csv'
        results_df = pd.read_csv(results_path)
        results_df = results_df.groupby('docid').unique_id.nunique().to_frame(name=f'{model}-{method}')
        relations_dfs.append(results_df)
        

# %%
result = pd.concat(relations_dfs, axis=1)
# %%
