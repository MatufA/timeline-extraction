# %%
from enum import unique
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as ET
import re

def replace_eid(text, exclude_ids):
    pattern = r'ei(\d+):(\w+)'
    
    def replace_func(match):
        ei_id = match.group(1)
        word = match.group(2)
        
        if f'ei{ei_id}' in exclude_ids:
            return f'[e{ei_id}]{word}[/e{ei_id}]'
        else:
            return word
    
    # Use re.sub with a replacement function
    result = re.sub(pattern, replace_func, text)
    return result


def parse_tml_to_dataframe(tml_file_path, docid):
    """
    Parses a TimeML (TML) file into a pandas DataFrame with the specified columns.

    Args:
        tml_file_path (str): Path to the TML file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['docid', 'plain_txt', 'eiid1', 'eiid2', 'relation'].
    """
    jsonl_format = []
    # Parse the TML file using ElementTree
    # tree = ET.parse(tml_file_path)
    # root = tree.getroot()
    with tml_file_path.open('r') as f:
            data = f.read()
    doc = BeautifulSoup(data, features='lxml')

    # docid = doc.find(name='docid').text
    

    # Extract plain text from the <TEXT> tag
    # plain_txt = doc.find(name='text').text.strip()
    mapping = {}
    for d in doc.find_all('makeinstance'):
        if d.get("eventid") not in mapping:
            mapping[d.get("eventid")] = d.get("eiid")
    
    text = doc.find(name="text")
    try:
        [d.replace_with(f'{mapping[d.get("eid")] if d.get("eid") in mapping else d.get("eid")}:{d.get_text()} ')  for d in text.find_all('event')]
        [d.replace_with(f'{d.get_text()} ') for d in text.find_all('timex3')]
        text = text.get_text().strip()
    except AttributeError:
        print(docid)
        print(text)
        raise

    events = set()
    unique_ids = set()

    # Extract TLINKs (temporal relations)
    tlinks = []
    for tlink in doc.find_all('tlink'):
        eiid1 = tlink.get('eventinstanceid', None)
        eiid2 = tlink.get('relatedtoeventinstance', None)
        relation = tlink.get('reltype', None)
        
        # Append only if all required attributes are present
        if eiid1 and eiid2 and relation:
            # relation = relation if relation != 'NONE' else 'VAGUE' 
            events.add(eiid1)
            events.add(eiid2)
            unique_id = '-'.join(sorted([eiid1, eiid2]))
            tlinks.append((docid, text, eiid1, eiid2, relation, unique_id))

            # formated_text = replace_eid(text, exclude_ids=[eiid1, eiid2])
            # lines = formated_text.split('\n')
            # start_idx = -1
            # end_idx = -1
            # for idx, line in enumerate(formated_text.split('\n')):
            #     if eiid1.replace('ei', 'e') in line:
            #         start_idx = idx
            #     if eiid2.replace('ei', 'e') in line:
            #         end_idx = idx

            if unique_id not in unique_ids:
                jsonl_format.append({
                    'text': text,
                    'relations': [(eiid1, eiid2)],
                    'docid': docid, 
                    'true_label': relation.lower() #if relation.lower() != 'NONE' else 'VAGUE' 
                })
                unique_ids.add(unique_id)

    # Create a DataFrame from the extracted TLINKs
    # print(f'{docid} - #events={len(events)} and #tlinks={len(tlinks)}')
    df = pd.DataFrame(tlinks, columns=['docid', 'text', 'eiid1', 'eiid2', 'relation', 'unique_id'])

    return df, len(events), len(tlinks), jsonl_format


# %%
# Example usage:
# Assuming you have a TML file at "example.tml" and want to assign it a docid "doc_001"
time_bank = Path('/home/adiel/nt/corpus/timebank/nt_converted_to_tml/a1')
datasets = []
total_events = []
total_tlinks = []
records_to_evaluate =[]

for file in time_bank.glob('*.tml'):
    df, n_events, n_tlinks, jsonl_format = parse_tml_to_dataframe(file, file.name.replace('.tml', ''))
    datasets.append(df)
    total_events.append(n_events)
    total_tlinks.append(n_tlinks)
    records_to_evaluate.extend(jsonl_format)

# %%
df_all = pd.concat(datasets, ignore_index=True)
# df_all.to_csv('/home/adiel/full-temporal-relation/data/TimeBank_dataset.csv', index=False)

# %% 
import json
from pathlib import Path

test_tml_files = [
    "APW19980227.0489",
    "APW19980227.0494",
    "APW19980308.0201",
    "APW19980418.0210",
    "CNN19980126.1600.1104",
    "CNN19980213.2130.0155",
    "NYT19980402.0453",
    "PRI19980115.2000.0186",
    "PRI19980306.2000.1675"
]
train_output = Path('/home/adiel/full-temporal-relation/data/narrativetime_a1_train.csv')
test_output = Path('/home/adiel/full-temporal-relation/data/narrativetime_a1_test.csv')

train_records = []
test_records = []
for record in records_to_evaluate:
    if record['docid'] in test_tml_files:
        test_records.append(record)
    else:
        train_records.append(record)

train_output.write_text(json.dumps(train_records, indent=2))
test_output.write_text(json.dumps(test_records, indent=2))
# %%
