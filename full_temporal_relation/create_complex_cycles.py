# %%
import pandas as pd

def join_cycles(cycles: list):
    cycles_together = []
    edges_sets = []

    for cycle in cycles:
        cycle_data = [(e1.split('-')[1], e2.split('-')[1], relation) for e1, e2, relation in cycle]
        current_edges = set()
        for e1, e2, _ in cycle_data:
            current_edges.add(e1)
            current_edges.add(e2)
        if edges_sets:
            for idx, edges in enumerate(edges_sets):
                if edges.intersection(current_edges):
                    cycles_together[idx].extend(cycle_data)
                    edges_sets[idx].update(current_edges - edges)
                else:
                    edges_sets.append(current_edges)
                    cycles_together.append(cycle_data)
        else:
            edges_sets.append(current_edges)
            cycles_together.append(cycle_data)

    cycle_dfs = []
    for idx, cycle in enumerate(cycles_together):
        cycle_df = pd.DataFrame(cycle, columns=['eiid1', 'eiid2', 'relation'])
        eiid1_eiid2 = list(zip(cycle_df['eiid1'], cycle_df['eiid2']))
        cycle_df['unique_id'] = ['-'.join(sorted([eiid1, eiid2])) for eiid1, eiid2 in eiid1_eiid2]
        cycle_df['cycle_id'] = idx
        cycle_dfs.append(cycle_df)
    return pd.concat(cycle_dfs)

# %%
from pathlib import Path
import json

output = Path('/home/adiel/full-temporal-relation/data/all_cycles_matres_by_baseline.json')
data = json.load(output.open('r'))
joints_cycles = {}
joints_cycles_dfs = []

for docid, cycles in data.items():
    print(f'{docid}: {len(cycles)}# cycles')
    # print(cycles)
    # break
    joint_cycles = join_cycles(cycles)
    joint_cycles['docid'] = docid
    joints_cycles_dfs.append(joint_cycles)
    # joints_cycles[docid] = joint_cycles
    print(f'{docid}: {len(joint_cycles)}# joint cycles')

# %%
df_all = pd.concat(joints_cycles_dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset=['docid', 'cycle_id', 'unique_id'])
# %%
df_all.to_csv('/home/adiel/full-temporal-relation/data/grouped_cycles_matres_by_baseline.csv', index=False)
# %%
