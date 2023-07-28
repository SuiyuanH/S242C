import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import os
from tqdm import tqdm

num_f = 10
num_case = 5 # num_pos / num_neg

srcpath = 'Results/Shap_all30_0725'
savepath = 'Results/Pairs_all30_0725'

os.makedirs(savepath, exist_ok=True)

sub_domains = ['reddit_eli5','finance','medicine','open_qa','wiki_csai','article']

all_data = pd.read_csv(f'{srcpath}/all_data_with_shap_0725.csv', encoding='utf-8')

# 读取features.csv
features_name = pd.read_csv('Results/Tables/features.csv', encoding='utf-8')

columns = ['domain', 'feature', 'feature_name', 'feature_rank', 'fpsp', 'fpsn', 'fnsp', 'fnsn', 'zero', 'direction', 'global_id', 'id', 'label', 'pred', 'proba', 'shap_value', 'feature_value', 'shap_value_rank', 'text']
case_df = pd.DataFrame(columns=columns)

fidxs = list (range (301))

shap_values = all_data.loc [:, [f'shap_{i}' for i in fidxs]].values
shap_values_abs = np.abs(shap_values)
shap_sum = np.mean(shap_values_abs, axis=0)
shap_rank = shap_sum.argsort () [::-1]

for frank in tqdm (range (num_f)):
    # fidx, fpsp, fpsn, fnsp, fnsn, zero, direction
    local_fidx = shap_rank [frank]
    fidx = fidxs [local_fidx]
    shap_value_all = shap_values [:, local_fidx]
    for domain in sub_domains:
        feature_value = all_data.loc [(all_data['domain'] == domain).values, f'feature_{fidx}']
        shap_value = shap_value_all [(all_data['domain'] == domain).values]
        feature_mean = np.median (feature_value)
        fpsp = ((shap_value > 0) & (feature_value > feature_mean)).sum ()
        fpsn = ((shap_value < 0) & (feature_value > feature_mean)).sum ()
        fnsp = ((shap_value > 0) & (feature_value < feature_mean)).sum ()
        fnsn = ((shap_value < 0) & (feature_value < feature_mean)).sum ()
        zero = ((shap_value == 0) | (feature_value == feature_mean)).sum ()
        if fpsp + fnsn > fpsn + fnsp:
            direction = 'same'
        elif fpsp + fnsn < fpsn + fnsp:
            direction = 'opposite'
        else:
            direction = 'null'
        ids = all_data.loc [(all_data['domain'] == domain) & (all_data['label'] == 'human'), 'id']
        ids = [i.split ('_') [0] for i in ids]
        pairs = [[f'{i}_human', f'{i}_machine'] for i in ids]
        scores = [[hid, mid, all_data.loc [(all_data['domain'] == domain) & (all_data['id'] == mid), f'shap_{fidx}'].values - all_data.loc [(all_data['domain'] == domain) & (all_data['id'] == hid), f'shap_{fidx}'].values] for hid, mid in pairs]
        scores = sorted (scores, key=lambda x: x[2], reverse=(direction == 'same'))
        for srank in range (num_case):
            hid, mid, score = scores [srank]
            hcidx = all_data.index [(all_data['domain'] == domain).values & (all_data['id'] == hid).values]
            mcidx = all_data.index [(all_data['domain'] == domain).values & (all_data['id'] == mid).values]
            case_df.loc [len (case_df)] = [domain, fidx, features_name.iloc [fidx, 0], frank, fpsp, fpsn, fnsp, fnsn, zero, direction, hcidx, all_data.iloc [hcidx, 0].values[0], all_data.iloc [hcidx, 5].values[0], all_data.iloc [hcidx, 7].values[0], all_data.iloc [hcidx, 6].values[0], all_data.loc [hcidx, f'shap_{fidx}'].values[0], all_data.loc [hcidx, f'feature_{fidx}'].values[0], srank, all_data.iloc [hcidx, 1].values[0]]
            case_df.loc [len (case_df)] = [domain, fidx, features_name.iloc [fidx, 0], frank, fpsp, fpsn, fnsp, fnsn, zero, direction, mcidx, all_data.iloc [mcidx, 0].values[0], all_data.iloc [mcidx, 5].values[0], all_data.iloc [mcidx, 7].values[0], all_data.iloc [mcidx, 6].values[0], all_data.loc [mcidx, f'shap_{fidx}'].values[0], all_data.loc [mcidx, f'feature_{fidx}'].values[0], srank, all_data.iloc [mcidx, 1].values[0]]

case_df.to_csv(f'{savepath}/pair_cases_0725.csv', index=False, encoding='utf-8')


