import sys
sys.path.append('src')
from influence_batch import IFEngine
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import json
import warnings
from tqdm import tqdm
import pandas as pd
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")

parser.add_argument("--tr_grad", type=str,default='', help='path for train_gradient')

parser.add_argument("--val_grad", type=str,default='', help='path for eval_gradient')

parser.add_argument("--tr_index", type=str,default='', help='index for batched gradient')

parser.add_argument('--save_all_layers', action='store_true', help='Enable saving all layers.')

parser.add_argument('--compute_lissa', action='store_true', help='compute with LiSSA(NIPS-17)')
# parser.add_argument("--val_index", type=str,default='', help='index for batched gradient')

parser.add_argument('--device',type=str,default='cuda:0', help='IF calculation device')

args = parser.parse_args()

def compute_val_grad_avg(val_grad_dict):
    # Compute the avg gradient on the validation dataset
    n_val = len(val_grad_dict)
    val_grad_avg_dict={}
    for weight_name in val_grad_dict[0]:
        if weight_name.__contains__('base_model'):
            val_grad_avg_dict[weight_name]=torch.zeros(val_grad_dict[0][weight_name].shape)
            for val_id in val_grad_dict:
                val_grad_avg_dict[weight_name] += val_grad_dict[val_id][weight_name] / n_val
        # else:
        #     val_grad_avg_dict[weight_name] = val_grad_dict[weight_name]
    # val_grad_avg_dict['ids'] = val_grad_dict['ids']
    return val_grad_avg_dict
def add_sum_column(df):
    """
    对输入的DataFrame添加一列，列的值为所有列名包含'model'的列的元素之和。

    参数:
    df (pd.DataFrame): 输入的DataFrame对象

    返回:
    pd.DataFrame: 添加了新列后的DataFrame对象
    """
    sum_cols = [col for col in df.columns if 'model' in col]
    df['sum_model_cols'] = df[sum_cols].sum(axis=1)
    return df

tr_grad_dict = torch.load(args.tr_grad)
val_grad_dict = torch.load(args.val_grad)

tr_index = torch.load(args.tr_index)
# val_index = torch.load(args.val_index)

IF_device = args.device

## 计算val_avg_grad_dict
val_grad_dict_avg = compute_val_grad_avg(val_grad_dict)

# print(tr_grad_dict)

for key in tqdm(tr_grad_dict.keys()): ## 放入GPU计算
    tr_grad_dict[key]['ids'] = tr_index[key]
    for kk in tr_grad_dict[key]:
        tr_grad_dict[key][kk] = tr_grad_dict[key][kk].to(IF_device)

for key in tqdm(val_grad_dict_avg):
        val_grad_dict_avg[key] = val_grad_dict_avg[key].to(IF_device)

influence_engine = IFEngine(weight_list=val_grad_dict_avg.keys())
influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict_avg)

influence_engine.compute_hvps(compute_accurate=False,compute_LiSSA=args.compute_lissa)
influence_engine.compute_IF()
result = influence_engine.IF_dict
sorted_dict_all = {}
for method in result.keys():
    result_df = pd.DataFrame(result[method]).T
    avg = add_sum_column(result_df) ## add all layers
    sample_IF = {}
    for index,row in avg.iterrows():
        ids = row['ids']
        sum = row['sum_model_cols']
        for id in ids:
            sample_IF[id] = sum
    sorted_dict = dict(sorted(sample_IF.items(), key=lambda x: x[1]))
    sorted_dict_all[method] = sorted_dict
    
    
# print(pd.DataFrame(result_df['proposed']))
# result_df.to_csv('grad/result_df.csv')
torch.save(sorted_dict_all,'Influence/AG-p2.pkl')
if args.save_all_layers:
    torch.save(result,'Influence/AG-p2-All-Layer.pkl')
            
