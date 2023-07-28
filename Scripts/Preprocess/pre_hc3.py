# 

import os 
import json
import numpy as np
import pandas  as pd

domains = ['reddit_eli5', 'finance', 'medicine', 'open_qa', 'wiki_csai']
model_name = 'HC3'
HC3_raw = './Raw_Data/HC3'
HC3_dst = './Data/data/full_text/HC3'
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

np.random.seed (175)

backward = {'id': [], 'text': [], 'label': [], 'source': [], 'domain': [], 'split': []}

for d in domains:
	d_raw = '{}/{}.jsonl'.format (HC3_raw, d)
	name_dst_h = '{}_human'.format (d)
	path = '{}/{}'.format (HC3_dst, name_dst_h)
	if not os.path.exists (path):
		os.makedirs (path)
	name_dst_c = '{}_chatgpt'.format (d)
	path = '{}/{}'.format (HC3_dst, name_dst_c)
	if not os.path.exists (path):
		os.makedirs (path)

	with open (d_raw, 'r', encoding='utf-8') as f:
		d_content = [json.loads (l.strip ()) for l in f.readlines () if l.strip ()]
		d_content = [(idx, i) for idx, i in enumerate(d_content) if i ['human_answers'] and i ['human_answers'][0] and i ['chatgpt_answers'] and i ['chatgpt_answers'][0]]
	num_samples = len (d_content)
	idxs = list (range (num_samples))
	np.random.shuffle (idxs)
	sep1 = int (train_ratio * num_samples)
	sep2 = int ((train_ratio + val_ratio) * num_samples)
	train_idxs = idxs [:sep1]
	valid_idxs = idxs [sep1:sep2]
	test_idxs = idxs [sep2:]

	d_human_train = []
	d_chat_train = []
	d_human_valid = []
	d_chat_valid = []
	d_human_test = []
	d_chat_test = []

	for idx, (cidx, c) in enumerate (d_content):
		backward ['id'].extend (['{}_human'.format (cidx), '{}_machine'.format (cidx)])
		backward ['text'].extend ([c ['human_answers'][0], c ['chatgpt_answers'][0]])
		backward ['label'].extend (['human', 'machine'])
		backward ['source'].extend (['HC3', 'HC3'])
		backward ['domain'].extend ([d, d])
		if idx in train_idxs:
			backward ['split'].extend (['train', 'train'])
			d_human_train.append (c ['human_answers'][0])
			if len (c ['human_answers']) > 1:
				print ('H warning: {}_{} has more human answers!'.format (d, idx))
			d_chat_train.append (c ['chatgpt_answers'][0])
			if len (c ['chatgpt_answers']) > 1:
				print ('H warning: {}_{} has more chatgpt answers!'.format (d, idx))
		elif idx in valid_idxs:
			backward ['split'].extend (['valid', 'valid'])
			d_human_valid.append (c ['human_answers'][0])
			if len (c ['human_answers']) > 1:
				print ('H warning: {}_{} has more human answers!'.format (d, idx))
			d_chat_valid.append (c ['chatgpt_answers'][0])
			if len (c ['chatgpt_answers']) > 1:
				print ('H warning: {}_{} has more chatgpt answers!'.format (d, idx))
		else:
			backward ['split'].extend (['test', 'test'])
			d_human_test.append (c ['human_answers'][0])
			if len (c ['human_answers']) > 1:
				print ('H warning: {}_{} has more human answers!'.format (d, idx))
			d_chat_test.append (c ['chatgpt_answers'][0])
			if len (c ['chatgpt_answers']) > 1:
				print ('H warning: {}_{} has more chatgpt answers!'.format (d, idx))

	path = '{}/{}/{}.train.jsonl'.format (HC3_dst, name_dst_h, name_dst_h)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (d_human_train, f, ensure_ascii=False)

	path = '{}/{}/{}.train.jsonl'.format (HC3_dst, name_dst_c, name_dst_c)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (d_chat_train, f, ensure_ascii=False)

	path = '{}/{}/{}.valid.jsonl'.format (HC3_dst, name_dst_h, name_dst_h)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (d_human_valid, f, ensure_ascii=False)

	path = '{}/{}/{}.valid.jsonl'.format (HC3_dst, name_dst_c, name_dst_c)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (d_chat_valid, f, ensure_ascii=False)

	path = '{}/{}/{}.test.jsonl'.format (HC3_dst, name_dst_h, name_dst_h)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (d_human_test, f, ensure_ascii=False)

	path = '{}/{}/{}.test.jsonl'.format (HC3_dst, name_dst_c, name_dst_c)
	with open (path, 'w', encoding='utf-8') as f:
		json.dump (d_chat_test, f, ensure_ascii=False)

pd.DataFrame (backward).to_csv (os.path.join (HC3_dst, 'HC3_dataset.csv'), encoding='utf-8')





