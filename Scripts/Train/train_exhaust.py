import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import xgboost as xgb

# 读取数据集作为all_data
all_data = pd.read_csv('Results/Tables/all_data_0723.csv', encoding='utf-8')

# 设置numpy的随机数种子为175
np.random.seed(175)

# 初始化记录表
res_csv = 'Tmp_Results/res.csv'
with open (res_csv, 'w', encoding='utf_8') as f:
    f.write ('domain,num_f,train_acc,valid_acc,test_acc,combination\n')

# 对于1: 300的常数num_f
for num_f in range(1, 301):
    # 对于'domain'定义的每一个子数据集
    for domain in all_data['domain'].unique():
        # 通过'domain'的取值从all_data中获取子数据集
        sub_data = all_data[all_data['domain'] == domain]
        
        # 通过'split'的取值划分训练集，验证集和测试集
        train_data = sub_data[sub_data['split'] == 'train']
        valid_data = sub_data[sub_data['split'] == 'valid']
        test_data = sub_data[sub_data['split'] == 'test']
        
        # 生成从300个特征中取num_f个的全部组合combs
        combs = list(itertools.combinations(['feature_' + str(i) for i in range(300)], num_f))
        
        # 对于combs中每一个组合comb
        print ('Processing {} with {} features...'.format (domain, num_f))
        for comb in tqdm (combs):
            # 使用训练集和comb所选的num_f个特征，对数据进行二分类为['human','machine']，模型使用xgboost
            X_train = train_data[list(comb)].values
            y_train = (train_data['label'].values == 'machine').astype (np.int32)
            X_valid = valid_data[list(comb)].values
            y_valid = (valid_data['label'].values == 'machine').astype (np.int32)
            X_test = test_data[list(comb)].values
            y_test = (test_data['label'].values == 'machine').astype (np.int32)
            
            # 训练xgboost模型
            model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
            model.fit(X_train, y_train)
            
            # 分别计算训练集，验证集和测试集上的准确率[train_acc, valid_acc, test_acc]
            train_acc = model.score(X_train, y_train)
            valid_acc = model.score(X_valid, y_valid)
            test_acc = model.score(X_test, y_test)
            
            # 更新记录表条目，记录domain, train_acc, valid_acc, test_acc, comb
            with open (res_csv, 'a', encoding='utf_8') as f:
                comb_pr = '_'.join ([i.split ('_')[-1] for i in comb])
                f.write (f'{domain},{num_f},{train_acc},{valid_acc},{test_acc},{comb_pr}\n')

print("Algorithm completed.")