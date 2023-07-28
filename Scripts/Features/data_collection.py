import os 
import json
import numpy as np
import pandas  as pd

args = [('HC3', 'finance_human', 'finance_chatgpt'),
        ('HC3', 'medicine_human', 'medicine_chatgpt'),
        ('HC3', 'open_qa_human', 'open_qa_chatgpt'),
        ('HC3', 'reddit_eli5_human', 'reddit_eli5_chatgpt'),
        ('HC3', 'wiki_csai_human', 'wiki_csai_chatgpt'),
        ('CGTD', 'CGTD_articles_human', 'CGTD_articles_chatgpt'),
        ('CGTD', 'CGTD_paragraphs_human', 'CGTD_paragraphs_chatgpt')]

HC3_prev = pd.read_csv ('./Data/data/full_text/HC3/HC3_dataset.csv', index_col=0, encoding='utf-8')
CGTD_articles_prev = pd.read_csv ('./Data/data/full_text/CGTD/CGTD_articles.csv', index_col=0, encoding='utf-8')
CGTD_paragraphs_prev = pd.read_csv ('./Data/data/full_text/CGTD/CGTD_paragraphs.csv', index_col=0, encoding='utf-8')

all_prev = pd.concat([HC3_prev, CGTD_articles_prev, CGTD_paragraphs_prev], axis=0)

new_columns = ['feature_' + str(i) for i in range(301)]
all_prev = all_prev.assign(**{column: 0 for column in new_columns})

for d in ['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai']:
    for l in ['human', 'chatgpt']:
        l1 = 'human' if l == 'human' else 'machine'
        for s in ['train', 'valid', 'test']:
            features = np.load('Data/features/full_text/HC3/{}_{}/{}_{}_features_{}.npy'.format (d, l, d, l, s), allow_pickle=True)
            if l == 'human':
                qs = np.load('Data/Q/full_text/HC3/{}_human\{}_human_{}_chatgpt_QFT_{}.npy'.format (d, d, d, s), allow_pickle=True)
            else:
                qs = np.load('Data/Q/full_text/HC3/{}_chatgpt\{}_chatgpt_QFT_{}.npy'.format (d, d, s), allow_pickle=True)
            fq = np.hstack ([features, qs])
            all_prev.loc[(all_prev['domain'] == d) & (all_prev['source'] == 'HC3') & (all_prev['split'] == s) & (all_prev['label'] == l1), new_columns] = fq

for d in ['articles', 'paragraphs']:
    d1 = 'article' if d == 'articles' else 'paragraph'
    for l in ['human', 'chatgpt']:
        l1 = 'human' if l == 'human' else 'machine'
        for s in ['train', 'valid', 'test']:
            features = np.load('Data/features/full_text/CGTD/CGTD_{}_{}/CGTD_{}_{}_features_{}.npy'.format (d, l, d, l, s), allow_pickle=True)
            if l == 'human':
                qs = np.load('Data/Q/full_text/CGTD/CGTD_{}_human\CGTD_{}_human_CGTD_{}_chatgpt_QFT_{}.npy'.format (d, d, d, s), allow_pickle=True)
            else:
                qs = np.load('Data/Q/full_text/CGTD/CGTD_{}_chatgpt\CGTD_{}_chatgpt_QFT_{}.npy'.format (d, d, s), allow_pickle=True)
            fq = np.hstack ([features, qs])
            all_prev.loc[(all_prev['domain'] == d1) & (all_prev['source'] == 'CGTD') & (all_prev['split'] == s) & (all_prev['label'] == l1), new_columns] = fq

all_prev.set_index (np.arange (len (all_prev)))
all_prev.to_csv ('./Results/Tables/all_data_0704.csv', encoding='utf-8')