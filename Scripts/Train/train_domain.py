import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import os

end = 297
#end = 301

num_f = 10
num_case = 5 # num_pos / num_neg

# 读取数据集作为all_data
all_data = pd.read_csv('Results/Tables/all_data_0723.csv', encoding='utf-8')
all_data [[f'shap_{i}' for i in range(301)]] = 0
all_data [['proba', 'pred_label']] = -1
all_data = all_data [['id', 'text', 'source', 'domain', 'split', 'label', 'proba', 'pred_label'] + [f'feature_{i}' for i in range(301)] + [f'shap_{i}' for i in range(301)]]

# 读取features.csv
features_name = pd.read_csv('Results/Tables/features.csv', encoding='utf-8')

# 设置numpy的随机数种子为175
np.random.seed(175)

# 初始化保存文件夹‘Result/Shap0724’=savepath
savepath = 'Results/Shap0724'
os.makedirs(savepath, exist_ok=True)

# 初始化记录表
columns = ['domain', 'train_acc', 'valid_acc', 'test_acc'] + [f'shap_{i}' for i in range(301)]
results_df = pd.DataFrame(columns=columns)

# 初始化案例表
columns = ['domain', 'feature', 'feature_name', 'feature_rank', 'fpsp', 'fpsn', 'fnsp', 'fnsn', 'zero', 'direction', 'global_id', 'id', 'label', 'pred', 'proba', 'shap_value', 'feature_value', 'shap_value_rank', 'text']
case_df = pd.DataFrame(columns=columns)

# 定义子数据集列表和split列表
domains = all_data['domain'].unique()  # 替换成您的子数据集名称列表
splits = ['train', 'valid', 'test']

lrs = [0.1, 0.03, 0.01, 0.003, 0.001]

for domain in domains:
    # 通过'domain'的取值从all_data中获取子数据集
    sub_data = all_data[all_data['domain'] == domain].copy()

    # 通过'split'的取值划分训练集，验证集和测试集
    data_split = sub_data[sub_data['split'] == 'train']

    # 使用训练集和全部特征，对数据进行二分类为['human','machine']，模型使用xgboost
    X = data_split [[f'feature_{i}' for i in range(end)]]
    y = (data_split ['label'] == 'machine').astype (np.int32)

    # 通过'split'的取值划分训练集，验证集和测试集
    data_split = sub_data[sub_data['split'] == 'valid']

    # 使用训练集和全部特征，对数据进行二分类为['human','machine']，模型使用xgboost
    Xv = data_split [[f'feature_{i}' for i in range(end)]]
    yv = (data_split ['label'] == 'machine').astype (np.int32)

    best_model = None
    best_acc = 0
    for lr in lrs:
        model = xgb.XGBClassifier(random_state=175)
        model.fit(X, y)

        accuracy = model.score(Xv, yv)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

    model = best_model
    accs = []

    for split in splits:
        # 通过'split'的取值划分训练集，验证集和测试集
        data_split = sub_data[sub_data['split'] == split]

        X = data_split [[f'feature_{i}' for i in range(end)]]
        y = (data_split ['label'] == 'machine').astype (np.int32)

        # 计算准确率
        accuracy = model.score(X, y)
        accs.append (accuracy)

    X = sub_data [[f'feature_{i}' for i in range(end)]]
    y = (sub_data ['label'] == 'machine').astype (np.int32)

    # 计算预测结果的概率和标签
    sub_data ['proba'] = model.predict_proba(X) [:, 1]
    sub_data ['pred_label'] = ['machine' if i else 'human' for i in model.predict(X).tolist ()]

    # 扩充all_data的表格
    all_data.update(sub_data)

    # 计算整个domain下所有数据点的shap值
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    all_data.loc [all_data['domain'] == domain, [f'shap_{i}' for i in range(end)]] = shap_values

    # 计算每个特征下shap值的绝对值和shap_sum，进行summary_plot，并保存到f'{{savepath}/shap_{domain}.png'
    shap_values_abs = np.abs(shap_values)
    shap_sum = np.mean(shap_values_abs, axis=0)
    filled = np.zeros (301)
    filled [:end] = shap_sum
    results_df.loc[len(results_df)] = [domain] + accs + filled.tolist ()

    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Summary Plot for {domain}')
    plt.savefig(f'{savepath}/shap_{domain}.png')
    plt.close()

    # 对于前若干个特征，找到一些案例
    shap_rank = shap_sum.argsort () [::-1]
    for frank in range (num_f):
        # fidx, fpsp, fpsn, fnsp, fnsn, zero, direction
        fidx = shap_rank [frank]
        shap_value = shap_values [:, fidx]
        feature_value = X.iloc [:, fidx].values
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
        shap_f_rank = shap_value.argsort () [::-1]
        # domain, global_id, id, label, pred, proba, shap_f, text
        for irank in range (5):
            cidx = shap_f_rank [irank]
            case_df.loc [len (case_df)] = [domain, fidx, features_name.iloc [fidx, 0], frank, fpsp, fpsn, fnsp, fnsn, zero, direction, sub_data.index [cidx], sub_data.iloc [cidx, 0], sub_data.iloc [cidx, 5], sub_data.iloc [cidx, 7], sub_data.iloc [cidx, 6], shap_value [cidx], feature_value [cidx], irank, sub_data.iloc [cidx, 1]]
        for irank in range (5):
            cidx = shap_f_rank [-1 - irank]
            case_df.loc [len (case_df)] = [domain, fidx, features_name.iloc [fidx, 0], frank, fpsp, fpsn, fnsp, fnsn, zero, direction, sub_data.index [cidx], sub_data.iloc [cidx, 0], sub_data.iloc [cidx, 5], sub_data.iloc [cidx, 7], sub_data.iloc [cidx, 6], shap_value [cidx], feature_value [cidx], -1 - irank, sub_data.iloc [cidx, 1]]


# 将更新后的all_data保存到f'{savepath}/all_data_with_shap_0724.csv'
all_data.to_csv(f'{savepath}/all_data_with_shap_0724.csv', index=False, encoding='utf-8')

# 将记录表保存到f'{savepath}/all_domains_with_shap_0724.csv'
results_df.to_csv(f'{savepath}/all_domains_with_shap_0724.csv', index=False, encoding='utf-8')

case_df.to_csv(f'{savepath}/all_cases_0724.csv', index=False, encoding='utf-8')
