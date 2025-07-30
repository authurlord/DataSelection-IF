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
# import submodlib
# from submodlib.functions.facilityLocation import FacilityLocationFunction
from datasets import Dataset
from src.load_dataset import ReTaskEvaluator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import yaml
import argparse
from src.evaluation import evaluation
evaluator = evaluation()
import torch.multiprocessing as mp
import subprocess
from src.util_func import load_yaml,z_score_normalize,get_top_k_indices,run_command,round_down_to_power_of_two,load_yaml_args

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type = str)
parser.add_argument('--device', type = str,default='4')
parser.add_argument('--model', type = str,default='mistral-7B')
parser.add_argument('--DO_SELECT', action='store_true')
parser.add_argument('--DO_TRAIN_MAIN', action='store_true')
parser.add_argument('--DO_EVAL_MAIN', action='store_true')
parser.add_argument('--DO_TRAIN', action='store_true')
parser.add_argument('--DO_EVAL', action='store_true')
parser.add_argument('--DO_TRAIN_IF_SINGLE', action='store_true')
parser.add_argument('--DO_EVAL_IF_SINGLE', action='store_true')
parser.add_argument('--DO_TRAIN_QURATING', action='store_true')
parser.add_argument('--DO_EVAL_QURATING', action='store_true')
input_args = parser.parse_args()

device = input_args.device
model = input_args.model
yaml_path = input_args.yaml_path
print(device,model,yaml_path)
### yaml path for config file

# device = '4'
# model = 'mistral-7B'

# yaml_path = 'script/config_DC_beer.yaml'

force_torchrun = '' ## FORCE_TORCHRUN=1 for single GPU in deepspeed
if len(device.split(','))==1:
    force_torchrun = 'FORCE_TORCHRUN=1'

# model = 'qwen2.5-7B'

cutoff_len = 1024
if model.lower().__contains__('qwen'):
    template_yaml_path = 'script/qwen2.5-7B_template.yaml' ## only for final training
elif  model.lower().__contains__('mistral'):
    template_yaml_path = 'script/mistral-7B_template.yaml' ## only for final training

# yaml_path = 'script/config_RE_RE.yaml'
# yaml_path = 'script/config_CTA_SimTab.yaml'
# yaml_path = 'script/config_ER_semi-text-w.yaml'




args = load_yaml(yaml_path)

task = args['task']
dataset = args['dataset']
os.makedirs('output/Ablation/{}/{}'.format(task,dataset),exist_ok=True)
DO_SELECT_ABLATION = input_args.DO_SELECT

DO_TRAIN = input_args.DO_TRAIN
DO_EVAL = input_args.DO_EVAL
DO_TRAIN_MAIN = input_args.DO_TRAIN_MAIN
DO_EVAL_MAIN = input_args.DO_EVAL_MAIN

DO_TRAIN_IF_SINGLE = input_args.DO_TRAIN_IF_SINGLE
DO_EVAL_IF_SINGLE = input_args.DO_EVAL_IF_SINGLE

DO_TRAIN_QURATING = input_args.DO_TRAIN_QURATING
DO_EVAL_QURATING = input_args.DO_EVAL_QURATING

if os.path.exists('eval_result/{}-{}-{}.npy'.format(model,task,dataset)):
    all_metrics = np.load('eval_result/{}-{}-{}.npy'.format(model,task,dataset),allow_pickle=True).item()
else:
    all_metrics = {}
print(task,all_metrics)
print(input_args)
print(args)
## load dataset

train_file_path = args['train_file_path']
train_file = pd.read_json(train_file_path)
select_file = pd.read_json('train/{}/{}/train-select.json'.format(task,dataset))
test_file_path = args['test_file_path']
## Load previous file for perplexity(PPL)
select_num = len(select_file) ## select size, can change

if DO_SELECT_ABLATION:
    ppl_array = np.zeros(len(train_file))
    for process_num in range(1,9,1): ## maximum of k process
        if os.path.exists('ppl/{}/{}/ppl-init-{}.csv'.format(task,dataset,process_num)): ## i-th gradient 
            ppl_df = pd.read_csv('ppl/{}/{}/ppl-init-{}.csv'.format(task,dataset,process_num),index_col=0)
            for index,row in ppl_df.iterrows():
                ppl_array[index] = row[0]
                
    ## Facility Location Score
    greedyList_All_norm_flatten = torch.load('selection/{}/{}/FL-Score.pkl'.format(task,dataset),weights_only=False)

    ## Calculation IF Score

    batch_sampler = torch.load('Influence/{}/{}/batch.pkl'.format(task,dataset),weights_only=False)
    sample_IF = torch.load('Influence/{}/{}/score.pkl'.format(task,dataset),weights_only=False)
    sample_IF = z_score_normalize(sample_IF) ## Normalize
    ## per-sample IF
    if os.path.exists('Influence_single/{}/{}'.format(task,dataset)):
        sample_IF_single = torch.load('Influence_single/{}/{}/score.pkl'.format(task,dataset),weights_only=False)
        for IF_method in sample_IF_single.keys():
            select_IF_single = get_top_k_indices(sample_IF_single[IF_method],k=select_num,IF=True)
            select_IF_single_df = train_file.iloc[select_IF_single]
            json.dump(select_IF_single_df.to_dict(orient='records'), open('train/{}/{}/train-select-w-IF-single-{}.json'.format(task,dataset,IF_method), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    ### Select by each component

    

    select_ppl = np.argsort(ppl_array)[-select_num:]

    select_FL = get_top_k_indices(greedyList_All_norm_flatten,k=select_num,IF=False)

    select_IF = get_top_k_indices(sample_IF['proposed'],k=select_num,IF=True)

    select_ppl_df = train_file.iloc[select_ppl]

    select_FL_df = train_file.iloc[select_FL]

    select_IF_df = train_file.iloc[select_IF]

    print(len(select_ppl_df),
        len(select_FL_df),
        len(select_IF_df)
        )

    json.dump(select_ppl_df.to_dict(orient='records'), open('train/{}/{}/train-select-w-ppl.json'.format(task,dataset), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    json.dump(select_FL_df.to_dict(orient='records'), open('train/{}/{}/train-select-w-FL.json'.format(task,dataset), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    json.dump(select_IF_df.to_dict(orient='records'), open('train/{}/{}/train-select-w-IF.json'.format(task,dataset), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

## DO_TRAIN



if DO_TRAIN:
    for ablation_method in ['ppl','FL','IF']:
        train_args = load_yaml(template_yaml_path)
        
        # if len(device.split(','))==1:
        #     train_args['use_unsloth'] = True
        #     del train_args['deepspeed']
        
        train_args['output_dir'] = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)
        
        train_args['train_file_path'] = 'train/{}/{}/train-select-w-{}.json'.format(task,dataset,ablation_method)
        
        train_args['cutoff_len'] = cutoff_len
        
        with open('script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(model,task,dataset,ablation_method), 'w') as file:
            yaml.dump(train_args, file)

        run_command('CUDA_VISIBLE_DEVICES={} {} llamafactory-cli train script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(
            device,
            force_torchrun,
            model,
            task,
            dataset,
            ablation_method)
                    )

### Inference

if DO_EVAL:
    # all_metrics = {}

    for ablation_method in ['ppl','FL','IF']:
        all_metrics[ablation_method] = {}
        lora_path = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)
        
        output_file_name = 'output/Ablation/{}/{}/{}-w-{}.csv'.format(task,dataset,model,ablation_method)
        
        print(round_down_to_power_of_two(len(device.split(','))))
        
        eval_command = 'CUDA_VISIBLE_DEVICES={} python vllm_query_qwen.py --lora_path {} --input_file {} --gpu_num {} --gpu_mem 0.8 --output_path {}'.format(
            device,
            lora_path,
            test_file_path,
            round_down_to_power_of_two(len(device.split(','))),
            output_file_name
        )
        
        run_command(eval_command)
        output_file = pd.read_csv(output_file_name,index_col=0)
        metrics = evaluator.process(task,dataset,output_file)
        all_metrics[ablation_method] = metrics


ablation_method = 'main'

train_args = load_yaml(template_yaml_path)

# if len(device.split(','))==1:
#     train_args['use_unsloth'] = True
#     del train_args['deepspeed']
train_args['output_dir'] = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)

train_args['train_file_path'] = 'train/{}/{}/train-select.json'.format(task,dataset)

train_args['cutoff_len'] = cutoff_len

with open('script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(
    model,
    task,
    dataset,
    ablation_method), 'w') as file:
    yaml.dump(train_args, file)

if DO_TRAIN_MAIN:
    run_command('CUDA_VISIBLE_DEVICES={} {} llamafactory-cli train script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(
        device,
        force_torchrun,
        model,
        task,
        dataset,
        ablation_method)
                )



if DO_EVAL_MAIN:
    lora_path = 'lora/{}/{}/{}/w-{}'.format(
        model,
        task,
        dataset,
        ablation_method
        )

    output_file_name = 'output/Ablation/{}/{}/{}-w-{}.csv'.format(task,dataset,model,ablation_method)
    
    eval_command = 'CUDA_VISIBLE_DEVICES={} python vllm_query_qwen.py --lora_path {} --input_file {} --gpu_num {} --gpu_mem 0.8 --output_path {}'.format(
        device,
        lora_path,
        test_file_path,
        round_down_to_power_of_two(len(device.split(','))),
        output_file_name
    )

    run_command(eval_command)

    output_file = pd.read_csv(output_file_name,index_col=0)

    metrics = evaluator.process(task,dataset,output_file)
    # if os.path.exists('eval_result/{}-{}-{}.npy'.format(model,task,dataset)):
    #     all_metrics = np.load('eval_result/{}-{}-{}.npy'.format(model,task,dataset),allow_pickle=True).item()
    # else:
    #     all_metrics = {}
    all_metrics[ablation_method] = metrics

# print(all_metrics)

if DO_TRAIN_IF_SINGLE:
    # for ablation_method in ['IF-single']:
    for ablation_method in ['IF-single-proposed']:
        train_args = load_yaml(template_yaml_path)
        
        train_args['output_dir'] = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)
        
        train_args['train_file_path'] = 'train/{}/{}/train-select-w-{}.json'.format(task,dataset,ablation_method)
        
        train_args['cutoff_len'] = cutoff_len
        
        with open('script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(model,task,dataset,ablation_method), 'w') as file:
            yaml.dump(train_args, file)

        train_command = 'CUDA_VISIBLE_DEVICES={} {} llamafactory-cli train script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(
            device,
            force_torchrun,
            model,
            task,
            dataset,
            ablation_method)
        
        print(train_command)
        run_command(train_command)

### Inference

if DO_EVAL_IF_SINGLE:
    # all_metrics = {}
    # os.makedirs('output/Ablation/{}/{}'.format(task,dataset),exist_ok=True)
    # for ablation_method in ['IF-single']:
    for ablation_method in ['IF-single-proposed']:
        all_metrics[ablation_method] = {}
        lora_path = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)
        
        output_file_name = 'output/Ablation/{}/{}/{}-w-{}.csv'.format(task,dataset,model,ablation_method)
        
        print(round_down_to_power_of_two(len(device.split(','))))
        
        eval_command = 'CUDA_VISIBLE_DEVICES={} python vllm_query_qwen.py --lora_path {} --input_file {} --gpu_num {} --gpu_mem 0.8 --output_path {}'.format(
            device,
            lora_path,
            test_file_path,
            round_down_to_power_of_two(len(device.split(','))),
            output_file_name
        )
        print(eval_command)
        run_command(eval_command)
        try:
            output_file = pd.read_csv(output_file_name,index_col=0)
            metrics = evaluator.process(task,dataset,output_file)
            # try:
            #     all_metrics[ablation_method] = metrics
            # except:
            all_metrics[ablation_method] = metrics
        except:
            print('IF not calculated, skip')
        # all_metrics[ablation_method] = metrics
        
### QuRating

if DO_TRAIN_QURATING:
    # for ablation_method in ['IF-single']:
    for ablation_method in ['QuRating']:
        try:
            select_QuRating = np.load('QuRating/data/select_index_main/{}-{}-QuRating.npy'.format(task,dataset))
            print('MAIN')
        except:
            select_QuRating = np.load('QuRating/data/select_index/{}-{}-QuRating.npy'.format(task,dataset)) 
            print('DSIR')        
        select_QuRating_df = train_file.iloc[select_QuRating]

        json.dump(select_QuRating_df.to_dict(orient='records'), open('train/{}/{}/train-select-w-{}.json'.format(task,dataset,ablation_method), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        
        train_args = load_yaml(template_yaml_path)
        
        train_args['output_dir'] = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)
        
        # train_args['num_train_epochs'] = 
        
        train_args['train_file_path'] = 'train/{}/{}/train-select-w-{}.json'.format(task,dataset,ablation_method)
        
        train_args['cutoff_len'] = cutoff_len
        
        with open('script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(model,task,dataset,ablation_method), 'w') as file:
            yaml.dump(train_args, file)

        train_command = 'CUDA_VISIBLE_DEVICES={} {} llamafactory-cli train script/ablation/{}_lora_{}_{}_P2-w-{}.yaml'.format(
            device,
            force_torchrun,
            model,
            task,
            dataset,
            ablation_method)
        
        print(train_command)
        run_command(train_command)

### Inference

if DO_EVAL_QURATING:
    # all_metrics = {}
    # os.makedirs('output/Ablation/{}/{}'.format(task,dataset),exist_ok=True)
    # for ablation_method in ['IF-single']:
    for ablation_method in ['QuRating']:
        all_metrics[ablation_method] = {}
        lora_path = 'lora/{}/{}/{}/w-{}'.format(model,task,dataset,ablation_method)
        
        output_file_name = 'output/Ablation/{}/{}/{}-w-{}.csv'.format(task,dataset,model,ablation_method)
        
        print(round_down_to_power_of_two(len(device.split(','))))
        
        eval_command = 'CUDA_VISIBLE_DEVICES={} python vllm_query_qwen.py --lora_path {} --input_file {} --gpu_num {} --gpu_mem 0.8 --output_path {}'.format(
            device,
            lora_path,
            test_file_path,
            round_down_to_power_of_two(len(device.split(','))),
            output_file_name
        )
        print(eval_command)
        run_command(eval_command)
        output_file = pd.read_csv(output_file_name,index_col=0)
        metrics = evaluator.process(task,dataset,output_file)
        # try:
        #     all_metrics[ablation_method] = metrics
        # except:
        all_metrics[ablation_method] = metrics
        # all_metrics[ablation_method] = metrics
print(metrics)
np.save('eval_result/{}-{}-{}.npy'.format(
    model,
    task,
    dataset
),all_metrics)