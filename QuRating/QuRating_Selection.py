import datasets
import pandas as pd
import numpy as np

import os
import shutil
import glob
import argparse
import subprocess
import yaml
# from src.util_func import load_yaml,run_command
import time
def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        args_dict = yaml.safe_load(file)
    return args_dict


def remove_dir_if_exists(path: str):
    """
    删除目标文件夹（如果存在）。
    :param path: 目标路径
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted folder: {path}")
    else:
        print(f"Folder does not exist: {path}")
def run_command(command):
    """
    执行给定的shell命令，并返回执行结果。
    """
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"命令 {e.cmd} 执行出错: {e.stderr.decode('utf-8')}")
        return None
def dataframe_to_jsonl_text_column(df: pd.DataFrame, output_filepath: str):
    """
    将 DataFrame 导出为 JSONL 文件，只包含 'instruction' 列，并将其重命名为 'text'。

    Args:
        df (pd.DataFrame): 待处理的 DataFrame。
        output_filepath (str): 输出 JSONL 文件的路径。
    """
    if 'instruction' not in df.columns:
        raise ValueError("DataFrame 必须包含 'instruction' 列。")

    # 选择 'instruction' 列并将其重命名为 'text'
    # 使用 .copy() 避免 SettingWithCopyWarning
    output_df = df[['instruction']].copy()
    output_df.rename(columns={'instruction': 'text'}, inplace=True)

    # 将 DataFrame 导出为 JSONL 格式
    # orient='records' 将每一行作为单独的 JSON 对象
    # lines=True 确保输出为 JSON Lines 格式
    # force_ascii=False 允许非 ASCII 字符（如中文）正常编码
    output_df.to_json(output_filepath, orient='records', lines=True, force_ascii=False)
    print(f"成功将 DataFrame 导出到 '{output_filepath}'，格式为 JSONL，包含 'text' 列。")

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str)
parser.add_argument('--dataset', type = str)
parser.add_argument('--select_size', type = int,default=0)
parser.add_argument('--device', type = str,default='6')
parser.add_argument('--model_path', type = str,default='../../model/QuRating-DSIR-1.3B')
parser.add_argument('--all',action='store_true')

args = parser.parse_args()

task = args.task
dataset = args.dataset
device = args.device
is_all_element = args.all
select_size = args.select_size
model_path = args.model_path ## path for pre-trained QuRating Model

## load config

yaml_path = '../script/config_{}_{}.yaml'.format(task,dataset)

config = load_yaml(yaml_path)
# select_config_path = '../script/mistral_lora_{}-{}_P2.yaml'.format(task,dataset)
# select_config = load_yaml(select_config_path)

train_file_path = config['train_file_path']
train_file = pd.read_json('../{}'.format(train_file_path))
train_file_size = len(train_file)

select_file_path = '../train/{}/{}/train-select.json'.format(task,dataset)
select_file = pd.read_json(select_file_path)
if select_size != 0:
    select_file_size = select_size
else:
    select_file_size = len(select_file)

print(task,dataset,train_file_size,select_file_size)

## Process Data

dataframe_to_jsonl_text_column(train_file, 'data/original/{}-{}-all.jsonl'.format(task, dataset))

## Select
start_time = time.time()

annotate_command = 'CUDA_VISIBLE_DEVICES={} python -m data_tools.qurater_annotate json data/annotation-main/{}-{} -F data/original/{}-{}-all.jsonl -M ../../model/QuRating-DSIR-1.3B --text_field text --labels writing_style required_expertise'.format(
    device,
    task,
    dataset,
    task,
    dataset)
print(annotate_command)
if not os.path.exists('data/annotation-main/{}-{}'.format(task,dataset)):
    output = run_command(annotate_command)
else:
    output = 'Task {} Dataset {} already embed, skip!'.format(task,dataset)
print(output)

annoate_time = time.time()

time_dict = {}

time_dict['annotate_time'] = annoate_time - start_time

## Calculate Average Token

annotate_data = datasets.load_from_disk('data/annotation-main/{}-{}/'.format(task,dataset))

token_count = []
for i in range(len(annotate_data)):
    token_count.append(annotate_data[i]['length'])
avg_token_count = np.mean(token_count)
if is_all_element:
    select_token_total = int(np.sum(token_count))
else:
    select_token_total = int(avg_token_count * select_file_size)

# select_token_total = 500_000

select_command = 'CUDA_VISIBLE_DEVICES={} python -m data_tools.select_subset data/annotation-main/{}-{}/ data/subset-main-writing/{}-{}/ \
    --metric_field writing_style_average \
    --seq_len_field length \
    --tokens {} \
    --temperature 2.0 \
    --normalize \
    --num_workers 1'.format(
        device,
        task,dataset,
        task,dataset,
        select_token_total
    )
remove_dir_if_exists(f'data/subset-main-writing/{task}-{dataset}/')
print(select_token_total)
print(select_command)
output = run_command(select_command)
print(output)
## load subset

subset = datasets.concatenate_datasets([datasets.load_from_disk(ds) for ds in sorted(glob.glob("data/subset-main-writing/{}-{}/*".format(task,dataset)))])

index_list = []
score = {}
for i in range(len(subset)):
    index = subset[i]['index']
    index_list.append(subset[i]['index'])
    score[index] = subset[i]['writing_style_average']
    
print('Original File Size:{}\n\nSelect File Size Target:{}\n\nSelect File Size Output:{}\n\nwith {} seconds'.format(train_file_size,select_file_size,len(index_list),time_dict['annotate_time']))
os.makedirs('data/select_index_main_writing',exist_ok=True)
if is_all_element:
    np.save('data/select_index_main_writing/{}-{}-QuRating.npy'.format(task,dataset),index_list)
    np.save('data/select_index_main_writing/{}-{}-QuRating-score.npy'.format(task,dataset),score)
elif select_size > 0: ## default size
    np.save('data/select_index_main_expert/{}-{}-QuRating-size-{}.npy'.format(task,dataset,select_file_size),index_list[:select_file_size])
else:
    np.save('data/select_index_main_expert/{}-{}-QuRating.npy'.format(task,dataset),index_list[:select_file_size])
np.save('data/select_index_main_expert/{}-{}-time-dict.npy'.format(task,dataset),time_dict)