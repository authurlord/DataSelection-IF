# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import json
import gc
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence
import yaml
import argparse
import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
import yaml
import argparse
import time
from llamafactory.data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer, get_dataset_id
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
import os
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
import os
from src.util_func import load_yaml


int_arg_list = ['cluster_num','sample_per_cluster','batch_size']
def create_folder_for_file(file_path):
    """
    给定一个文件的路径，判断路径中的文件夹是否存在，不存在则创建。

    参数:
    file_path (str): 文件的完整路径，例如 "A/B/C/df.csv"

    返回:
    None
    """
    # 获取文件所在目录的路径（去除文件名部分）
    folder_path = os.path.dirname(file_path)
    current_path = ""
    # 分割路径字符串为各个文件夹名组成的列表
    folders = folder_path.split(os.path.sep)
    for folder in folders:
        current_path = os.path.join(current_path, folder)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
def generate_list(n, batch_size):
    """
    根据给定的正整数n和batch_size，生成一个列表，列表中每个元素（子列表）长度尽量为batch_size，
    最后一个子列表长度可小于batch_size。

    参数:
    n (int): 总元素个数
    batch_size (int): 每个子列表期望的长度

    返回:
    list: 符合要求的子列表组成的列表
    """
    result = []
    full_batches = n // batch_size
    remainder = n % batch_size
    for i in range(full_batches):
        sublist = list(range(i * batch_size, (i + 1) * batch_size))
        result.append(sublist)
    if remainder > 0:
        last_sublist = list(range(full_batches * batch_size, full_batches * batch_size + remainder))
        result.append(last_sublist)
    return result
def load_yaml_args(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        args_dict = yaml.safe_load(file)
    # print(args_dict)
    parser = argparse.ArgumentParser()
    for key, value in args_dict.items():
        if isinstance(value, bool):
            # 将arg3指定为存储布尔值的参数，action='store_true'表示如果参数出现则为True
            parser.add_argument(f'--{key}', action='store_true', default = value)
        elif key in int_arg_list:
            parser.add_argument(f'--{key}', type = int, default = int(value))
        else:
            parser.add_argument(f'--{key}', default = value)
    parser.add_argument('--yaml_path', type = str)
    parser.add_argument('--process_num', type = int,default=1)
    parser.add_argument('--total_process_num', type = int,default=1)
    parser.add_argument('--IF_device', type = str,choices=['cuda','cpu'],default='cuda')
    parser.add_argument(
        '--use_main_model',
        action='store_true',
        help='如果指定此参数，表示使用主模型。'
    )
    args = parser.parse_args()
    if args.use_main_model: ## overwrite with 7B model
        config_main = load_yaml('script/mistral_lora_{}_{}_P2.yaml'.format(args.task,args.dataset))
        args.model_name_or_path = config_main['model_name_or_path']
        args.adapter_name_or_path = config_main['output_dir']
        args.template = config_main['template']
    return args

def split_list(input_list, n):
    """
    将给定的列表划分为n个元素数量大体相当的子列表。

    参数:
    input_list (list): 需要划分的列表
    n (int): 要划分成的子列表个数

    返回:
    list: 划分后的n个子列表组成的列表
    """
    list_length = len(input_list)
    if n <= 0:
        raise ValueError("n必须大于0")
    if list_length == 0:
        return [[] for _ in range(n)]
    base_size = list_length // n
    remainder = list_length % n
    result = []
    start_index = 0
    for i in range(n):
        sublist_size = base_size + (1 if remainder > 0 else 0)
        sublist = input_list[start_index:start_index + sublist_size]
        result.append(sublist)
        start_index += sublist_size
        remainder -= 1
    return result
def create_folder_for_file(file_path):
    """
    给定一个文件的路径，判断路径中的文件夹是否存在，不存在则创建。

    参数:
    file_path (str): 文件的完整路径，例如 "A/B/C/df.csv"

    返回:
    None
    """
    # 获取文件所在目录的路径（去除文件名部分）
    folder_path = os.path.dirname(file_path)
    current_path = ""
    # 分割路径字符串为各个文件夹名组成的列表
    folders = folder_path.split(os.path.sep)
    for folder in folders:
        current_path = os.path.join(current_path, folder)
        if not os.path.exists(current_path):
            os.mkdir(current_path)

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
class CustomBatchSampler:
    def __init__(self, batch_list):
        self.batch_list = batch_list

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self):
        for batch in self.batch_list:
            yield batch
@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    train_on_prompt: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.
        """
        chosen_features = []
        for feature in features:
            chosen_features.append(
                {
                    "input_ids": feature["chosen_input_ids"],
                    "attention_mask": feature["chosen_attention_mask"],
                    "labels": feature["chosen_input_ids"] if self.train_on_prompt else feature["chosen_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
            )

        return super().__call__(chosen_features)


def calculate_ppl(
    model_name_or_path: str = None,
    adapter_name_or_path: str = None,
    save_name: str = "ppl.json",
    batch_size: int = 1,
    stage: Literal["pt", "sft", "rm"] = "sft",
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "/data/home/wangys/LLAMA-backup/LLaMA-Factory/data",
    template: str = "default",
    cutoff_len: int = 1024,
    max_samples: Optional[int] = None,
    train_on_prompt: bool = False,
    tokenized_path: str = None,
    train_file_path: str = None,
    eval_file_path: str = None,
    grad_name :str = 'grad.pkl',
    process_num: int = 1,
    total_process_num : int = 1,
    batch_divide_path: str = None,
    yaml_path : str = None
):
    r"""
    Calculates the ppl on the dataset of the pre-trained models.
    Usage: export CUDA_VISIBLE_DEVICES=0
    python cal_ppl.py --model_name_or_path path_to_model --dataset alpaca_en_demo --save_name ppl.json
    """
    print(yaml_path)
    start_time = time.time()
    args = load_yaml_args(yaml_path)
    print(args)
    task = args.task
    dataset_name = args.dataset
    train_file = pd.read_json(args.train_file_path)
    batch_single = []
    for i in  range(len(train_file)):
        batch_single.append([i])
    batch_divide_path = 'Influence/{}/{}/batch_single.pkl'.format(task,dataset_name)
    torch.save(batch_single,batch_divide_path)

    
    model_name_or_path = args.model_name_or_path
    adapter_name_or_path = args.adapter_name_or_path
    train_file_path = args.train_file_path
    eval_file_path = args.eval_file_path
    stage = args.stage
    template = args.template
    
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage=stage,
            model_name_or_path=model_name_or_path,
            adapter_name_or_path = adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            tokenized_path = tokenized_path,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            train_on_prompt=train_on_prompt,
            output_dir="dummy_dir",
            overwrite_cache=True,
            do_train=False,
            train_file_path = train_file_path,
            eval_file_path = eval_file_path
        )
    )
    batch_divide_path = 'Influence/{}/{}/batch_single.pkl'.format(task,dataset_name)
    batch_list = torch.load(batch_divide_path) ## FL-based数据划分
    # batch_list = [batch for batch in batch_list if len(batch)>1] ## 过滤空batch和长度只为1的batch
    
    sub_batch_list = split_list(batch_list,total_process_num)[process_num-1] ## 多线程
    

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # trainset = get_dataset(template, model_args, data_args, training_args, stage, **tokenizer_module)["train_dataset"]
    
    if stage == "pt":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif stage == "sft":
        data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template, tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX
        )
    elif stage == "rm":
        data_collator = PairwiseDataCollatorWithPadding(
            template=template, tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX, train_on_prompt=train_on_prompt
        )
    else:
        raise NotImplementedError(f"Stage does not supported: {stage}.")


    trainset = get_dataset_id(template, model_args, data_args, training_args, stage, **tokenizer_module)["train_dataset"]
    ### Temporary modify data_args.train_file_path to data_args.eval_file_path
    train_file_path = data_args.train_file_path
    if (data_args.eval_file_path is not None): ## eval_file_path exists
        data_args.train_file_path = data_args.eval_file_path
        eval_dataset = get_dataset_id(template, model_args, data_args, training_args, stage, **tokenizer_module)["train_dataset"]
        data_args.train_file_path = train_file_path

    
    
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
    # print('sub_batch_list \n\n')
    # print(sub_batch_list)
    batch_sampler = CustomBatchSampler(sub_batch_list)
    sub_batch_list_eval = split_list(generate_list(len(eval_dataset),batch_size),total_process_num)[process_num-1] ## 多线程
    eval_batch_sampler = CustomBatchSampler(sub_batch_list_eval)
    
    dataloader = DataLoader(trainset, collate_fn=data_collator, pin_memory=True,batch_sampler=batch_sampler)
    
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, pin_memory=True,batch_sampler=eval_batch_sampler)
    
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_ppl = 0
    perplexities = []
    batch: Dict[str, "torch.Tensor"]
    # with torch.no_grad():
    tr_grad_dict = {}
    val_grad_dict = {}
    model.eval()
    # torch.set_grad_enabled(True)
    id_list = [] ## id list in batch for mapping
    id_list_flatten = []
    ## 迭代Train_Data
    for step,batch in enumerate(tqdm(dataloader)):
        # print(batch.keys())
        # print(batch['ids'])
        id_list.append(batch['ids'])
        id_list_flatten.extend(batch['ids'])
        # print(batch['ids'],batch_list[step])
        model.zero_grad()
        batch = batch.to(model.device)
        outputs = model(**batch)
        loss = outputs.loss

        shift_logits: "torch.Tensor" = outputs["logits"][..., :-1, :]
        shift_labels: "torch.Tensor" = batch["labels"][..., 1:]
        loss_mask = shift_labels != IGNORE_INDEX
        flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
        flatten_labels = shift_labels.contiguous().view(-1)
        token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels)
        token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
        sentence_logps = (token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        total_ppl += sentence_logps.exp().sum().item()
        perplexities.extend(sentence_logps.exp().tolist())
        ### IF
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        grad_dict={}
        for k, v in model.named_parameters():
            # print(k)
            if 'lora_A' in k:
                grad_dict[k]=v.grad.cpu()
            elif 'lora_B' in k:
                # first index of shape indicates low-rank
                grad_dict[k]=v.grad.cpu().T
            else:
                pass
        # for k, v in model.named_parameters():
        #     if 'lora_A' in k:
        #         # 确保 grad 不为 None，并且 detach 以防止其被后续计算追踪
        #         if v.grad is not None:
        #             grad_dict[k] = v.grad.cpu().detach().clone()
        #     elif 'lora_B' in k:
        #         if v.grad is not None:
        #             grad_dict[k] = v.grad.cpu().detach().clone().T
        #         else:
        #             pass
        tr_grad_dict[step]=grad_dict
        # ## remove non-used variables
        # torch.cuda.empty_cache()
        # gc.collect()
    tr_index = id_list


    # with open(save_name, "w", encoding="utf-8") as f:
    #     json.dump(perplexities, f, indent=2)
    # perplexities_df = pd.DataFrame(perplexities)
    # perplexities_df.index = [int(x) for x in id_list_flatten]
    # print([int(x) for x in id_list_flatten])
    # create_folder_for_file('ppl_single/{}/{}/ppl-p2-{}.csv'.format(task,dataset_name,process_num))
    # create_folder_for_file('grad_single/{}/{}/tr_grad_{}.pkl'.format(task,dataset_name,process_num))
    # perplexities_df.to_csv('ppl_single/{}/{}/ppl-p2-{}.csv'.format(task,dataset_name,process_num))
    # torch.save(tr_grad_dict,'grad_single/{}/{}/tr_grad_{}.pkl'.format(task,dataset_name,process_num))
    # torch.save(id_list,'grad_single/{}/{}/tr_grad_index_{}.pkl'.format(task,dataset_name,process_num))
    # print(f"Average perplexity is {total_ppl / len(perplexities):.2f}")
    # print(f"Perplexities have been saved at {save_name}.")

## Clear the id_list, and calculate eval_dataloader if possible
    if (eval_file_path is not None):
        id_list = [] ## id list in batch for mapping
        id_list_flatten = []
        total_ppl = 0
        perplexities = []
        
        for step,batch in enumerate(tqdm(eval_dataloader)):
            # print(batch.keys())
            id_list.append(batch['ids'])
            id_list_flatten.extend(batch['ids'])
            # print(batch['ids'],batch_list[step])
            model.zero_grad()
            batch = batch.to(model.device)
            outputs = model(**batch)
            loss = outputs.loss

            shift_logits: "torch.Tensor" = outputs["logits"][..., :-1, :]
            shift_labels: "torch.Tensor" = batch["labels"][..., 1:]
            loss_mask = shift_labels != IGNORE_INDEX
            flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
            flatten_labels = shift_labels.contiguous().view(-1)
            token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels)
            token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
            sentence_logps = (token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            total_ppl += sentence_logps.exp().sum().item()
            perplexities.extend(sentence_logps.exp().tolist())
            ### IF
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            grad_dict={}
            for k, v in model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            val_grad_dict[step]=grad_dict
        # perplexities_df = pd.DataFrame(perplexities)
        # perplexities_df.index = [int(x) for x in id_list_flatten]
        # perplexities_df.to_csv('ppl_single/{}/{}/ppl-p2-{}-eval.csv'.format(task,dataset_name,process_num))
        # torch.save(val_grad_dict,'grad_single/{}/{}/val_grad_{}.pkl'.format(task,dataset_name,process_num))
        # torch.save(id_list,'grad_single/{}/{}/val_grad_index_{}.pkl'.format(task,dataset_name,process_num))
    grad_time = time.time()
        ## Calculate IF Score from here, without i/o process
    val_grad_dict_avg = compute_val_grad_avg(val_grad_dict)
    for key in tqdm(tr_grad_dict.keys()): ## 放入GPU计算
        tr_grad_dict[key]['ids'] = tr_index[key]
        for kk in tr_grad_dict[key]:
            tr_grad_dict[key][kk] = tr_grad_dict[key][kk].to(args.IF_device)

    for key in tqdm(val_grad_dict_avg):
            val_grad_dict_avg[key] = val_grad_dict_avg[key].to(args.IF_device)
            
    influence_engine = IFEngine(weight_list=val_grad_dict_avg.keys())
    influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict_avg)

    influence_engine.compute_hvps(compute_accurate=False,compute_LiSSA=False)
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
    IF_time = time.time()
    time_dict = {}
    time_dict['gradient-calculation'] = grad_time - start_time
    time_dict['IF-Score'] = IF_time - grad_time
    
    # print(pd.DataFrame(result_df['proposed']))
    os.makedirs('Influence_single/{}/{}'.format(task,dataset_name),exist_ok=True)


    result_df.to_csv('Influence_single/{}/{}/result_df.csv'.format(task,dataset_name))

    create_folder_for_file('Influence_single/{}/{}/result.pkl'.format(task,dataset_name))
    torch.save(sorted_dict_all,'Influence_single/{}/{}/score.pkl'.format(task,dataset_name))
    # if args.save_all_layers:
    torch.save(result,'Influence_single/{}/{}/result.pkl'.format(task,dataset_name))
    np.save('Influence_single/{}/{}/time.npy'.format(task,dataset_name),time_dict)
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--yaml_path', type = str)
    # parser.add_argument('--process_num', type = int,default=1)
    # parser.add_argument('--total_process_num', type = int,default=1)
    # args = parser.parse_args()
    fire.Fire(calculate_ppl)
    # calculate_ppl(yaml_path=args.yaml_path, process_num=args.process_num,total_process_num=args.total_process_num)
