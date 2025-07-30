import numpy as np
from sklearn.cluster import  KMeans
from sklearn.manifold import TSNE
import torch
import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
from FlagEmbedding import FlagModel
import time
import submodlib
from submodlib.functions.facilityLocation import FacilityLocationFunction
from datasets import Dataset

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import yaml
import argparse

import torch.multiprocessing as mp
import subprocess
from src.util_func import run_command,load_yaml_args

# yaml_path = 'script/config_ER_semi-text-w.yaml'
# yaml_path = 'script/config_RE_RE.yaml'
yaml_path = 'script/config_DC_beer.yaml'
args = load_yaml_args(yaml_path)
train_file = pd.read_json(args.train_file_path)
device = '4'
device_list = device.split(',')

time_dict = {}

command_IF = []
start_time = time.time()

DO_CAL_GRAD = False
DO_CAL_IF = True

batch_single = []
for i in  range(len(train_file)):
    batch_single.append([i])
# batch_single
task = args.task
dataset_name = args.dataset
batch_divide_path = 'Influence/{}/{}/batch_single.pkl'.format(task,dataset_name)
torch.save(batch_single,batch_divide_path)

for process_num in range(len(device_list)): ## 从0开始
    command_IF.append('CUDA_VISIBLE_DEVICES={} python cal_IF_single.py --yaml_path {} --process_num {} --total_process_num {}'.format(device_list[process_num],yaml_path,process_num+1,len(device_list)))

# if not hasattr(mp, '_start_method'):
#     mp.set_start_method('spawn')
processes = []
if DO_CAL_GRAD:
    for command in command_IF:
        p = mp.Process(target=run_command, args=(command,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("所有命令执行完毕，进行下一步")

end_time = time.time()
time_dict['gradient-calculation'] = end_time - start_time


if DO_CAL_IF:
    
    print('CUDA_VISIBLE_DEVICES={} python IF_Cal_single.py  --device cuda --task {} --dataset {} --save_all_layers --compute_lissa'.format(device,args.task,args.dataset))

    run_command('CUDA_VISIBLE_DEVICES={} python IF_Cal_single.py  --device cuda --task {} --dataset {} --save_all_layers --compute_lissa'.format(device,args.task,args.dataset))

end_time = time.time()
time_dict['IF-Score'] = end_time - start_time

print(time_dict)