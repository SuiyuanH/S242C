# 

import os 
import json
import numpy as np
import pandas as pd

domains = ['human', 'chatgpt']
CGTD_raw = './Raw_Data/chatgpt-generated-text-detection-corpus-main/full_texts'
CGTD_dst = './Data/data/full_text/CGTD'
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

np.random.seed (175)

backward = {'id': [], 'text': [], 'label': [], 'source': [], 'domain': [], 'split': []}
backward2 = {'id': [], 'text': [], 'label': [], 'source': [], 'domain': [], 'split': []}

for d in domains:
	label = 'human' if d == 'human' else 'machine'
	rids = [int (name.split ('.')[0]) for name in os.listdir ('{}/{}'.format (CGTD_raw, d))]
	rids.sort ()
	num_samples = len (rids)
	idxs = list (range (num_samples))
	sep1 = int (train_ratio * num_samples)
	sep2 = int ((train_ratio + val_ratio) * num_samples)
	idxs_train = idxs [:sep1]
	idxs_valid = idxs [sep1: sep2]
	idxs_test = idxs [sep2:]

	path_a = '{}/CGTD_articles_{}'.format (CGTD_dst, d)
	if not os.path.exists (path_a):
		os.makedirs (path_a)
	path_p = '{}/CGTD_paragraphs_{}'.format (CGTD_dst, d)
	if not os.path.exists (path_p):
		os.makedirs (path_p)
	
	articles_train = []
	articles_valid = []
	articles_test = []
	paragraphs_train = []
	paragraphs_valid = []
	paragraphs_test = []

	for idx, rid in enumerate (rids):
		file = '{}/{}/{}.txt'.format (CGTD_raw, d, rid)
		with open (file, 'r', encoding='utf-8') as f:
			content = f.read ()
		text = content.strip ()
		if text:
			backward ['id'].append ('{}_{}'.format (rid, label))
			backward ['text'].append (text)
			backward ['label'].append (label)
			backward ['source'].append ('CGTD')
			backward ['domain'].append ('article')
			ps = [i.strip () for i in content.split ('\n') if i.strip ()]
			lps = len (ps)
			backward2 ['id'].extend (['{}_{}'.format (rid, sid) for sid in range (lps)])
			backward2 ['text'].extend (ps)
			backward2 ['label'].extend ([label for sid in range (lps)])
			backward2 ['source'].extend (['CGTD' for sid in range (lps)])
			backward2 ['domain'].extend (['paragraph' for sid in range (lps)])
			if idx in idxs_train:
				articles_train.append (text)
				paragraphs_train.extend (ps)
				backward ['split'].append ('train')
				backward2 ['split'].extend (['train' for sid in range (lps)])
			elif idx in idxs_valid:
				articles_valid.append (text)
				paragraphs_valid.extend (ps)
				backward ['split'].append ('valid')
				backward2 ['split'].extend (['valid' for sid in range (lps)])
			else:
				articles_test.append (text)
				paragraphs_test.extend (ps)
				backward ['split'].append ('test')
				backward2 ['split'].extend (['test' for sid in range (lps)])


	path = '{}/CGTD_articles_{}.train.jsonl'.format (path_a, d)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (articles_train, f, ensure_ascii=False)
	path = '{}/CGTD_paragraphs_{}.train.jsonl'.format (path_p, d)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (paragraphs_train, f, ensure_ascii=False)
    
	path = '{}/CGTD_articles_{}.valid.jsonl'.format (path_a, d)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (articles_valid, f, ensure_ascii=False)
	path = '{}/CGTD_paragraphs_{}.valid.jsonl'.format (path_p, d)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (paragraphs_valid, f, ensure_ascii=False)

	path = '{}/CGTD_articles_{}.test.jsonl'.format (path_a, d)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (articles_test, f, ensure_ascii=False)
	path = '{}/CGTD_paragraphs_{}.test.jsonl'.format (path_p, d)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (paragraphs_test, f, ensure_ascii=False)
	
pd.DataFrame (backward).to_csv (os.path.join (CGTD_dst, 'CGTD_articles.csv'), encoding='utf-8')
pd.DataFrame (backward2).to_csv (os.path.join (CGTD_dst, 'CGTD_paragraphs.csv'), encoding='utf-8')