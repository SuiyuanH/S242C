import os 
import json
import numpy as np
import pandas as pd

from demo_indicating_words import auto_filter

hc3_df = pd.read_csv ('Data/data/full_text/HC3/HC3_dataset.csv', encoding='utf-8')
cgtd_df = pd.read_csv ('Data/data/full_text/CGTD/CGTD_articles.csv', encoding='utf-8')

base_csv = pd.concat ([hc3_df, cgtd_df], axis=0)
base_csv.index = np.arange (len (base_csv))

unique_domain = np.unique (base_csv ['domain'].values)
unique_label = ['human', 'machine']
unique_split = np.unique (base_csv ['split'].values)

# 创建一个元素全部为[]的Dict
base_dict = {domain: {label: {split: [] for split in unique_split} for label in unique_label} for domain in unique_domain}

for i in range (len (base_csv)):
    line = base_csv.loc [i, :]
    text = line ['text']
    label = line ['label']
    domain = line ['domain']
    split = line ['split']
    new_text = auto_filter (text, label == 'machine')
    base_dict [domain][label][split].append (new_text)
    base_csv.loc [i, 'text'] = new_text

for domain in unique_domain:
    for label in unique_label:
        label_path = f'Data/data/full_text/base/{domain}_{label}'
        os.makedirs (label_path, exist_ok=True)
        for split in unique_split:
            with open (f'{label_path}/{domain}_{label}.{split}.jsonl', 'w', encoding='utf-8') as f:
                json.dump (base_dict [domain][label][split], f)
            
base_csv.to_csv ('Data/data/full_text/base/base_dataset.csv', encoding='utf-8')

