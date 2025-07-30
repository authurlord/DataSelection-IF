import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import torch
from matplotlib import pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

## set seed for production
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class MyDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.index = 0  # 用于记录整体的索引

    def __next__(self):
        if self.index >= len(self.dataset):
            self.index = 0
            raise StopIteration
        batch_data = []
        batch_indices = []
        for _ in range(self.batch_size):
            if self.index < len(self.dataset):
                batch_data.append(self.dataset[self.index])
                batch_indices.append(self.index)
                self.index += 1
        collated_data = self.collate_fn(batch_data)
        collated_data['indices'] = torch.tensor(batch_indices)
        return collated_data

task_to_keys = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "ER": ("sentence_1", "sentence_2"), ## Add task name from here
    "WebTable": ("sentence_1", "sentence_2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
}

def flip_label(example, ind, noise_index):
    if ind in noise_index:
        example["label"] = 1 - example["label"]
    return example
id_counters = {
    "train": 0,
    "validation": 0,
    "test": 0
}

def add_id(example, split):
    """
    定义一个函数用于给单个样本添加id字段，根据传入的数据集划分名称来确定对应的id序号
    """
    global id_counters
    example["id"] = id_counters[split]
    id_counters[split] += 1
    return example
def load_noisy_dataset_by_task(task="mrpc", noise_ratio=0):
    # glue_datasets = load_dataset("/home/yanmy/DataInf/datasets/GLUE", task) 
    # glue_datasets = load_dataset("/home/yanmy/DataInf/datasets/GLUE/mrpc")
    glue_datasets = load_dataset("../glue/{}".format(task))


    # for split in ["train", "validation", "test"]:
    #     glue_datasets[split] = glue_datasets[split].map(add_id)
    for split in ["train", "validation", "test"]:
        glue_datasets[split] = glue_datasets[split].map(lambda x: add_id(x, split))
    # print(glue_datasets)
    n_train = len(glue_datasets['train'])
    n_val = len(glue_datasets['validation'])
    n_test = len(glue_datasets['test'])
    # print(glue_datasets['test'])
    # print(n_train,n_val,n_test)
    # if n_train > 4500 and n_val > 500:
    #     new_n_train_list = np.random.choice(n_train, 7000, replace=False)
    #     # new_n_val_list = np.random.choice(n_val, 500, replace=False)
    #     glue_datasets['train'] = glue_datasets['train'].select(new_n_train_list)
    #     # glue_datasets['validation'] = glue_datasets['validation'].select(new_n_val_list)
    
    n_train = len(glue_datasets['train'])
    n_val = len(glue_datasets['validation'])
    # if noise_ratio > 0.0:
    #     noise_index = np.random.choice(n_train,
    #                                    size=int(noise_ratio*n_train),
    #                                    replace=False)
    # else:
    noise_index = []
    
    # glue_datasets['train'] = glue_datasets['train'].map(flip_label, 
    #                                                     with_indices=True,
    #                                                     fn_kwargs={'noise_index':noise_index})
    return glue_datasets, noise_index

def create_dataloaders(model_name_or_path="roberta-large",
                       task="mrpc",
                       noise_ratio=0.1,
                       batch_size=32,
                       max_length=128):
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right") ## for deberta
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding=True,truncation=True)  ## for bert


    sentence1_key, sentence2_key = task_to_keys[task]
    def tokenize_function(examples, max_length=max_length):
        # max_length=None => use the model max length (it's actually the default)
        if sentence2_key is None:
            outputs = tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
        else:
            outputs = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=max_length)
        return outputs

    noisy_datasets, noise_index=load_noisy_dataset_by_task(task=task, noise_ratio=noise_ratio)
    if sentence2_key is None:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key],
        )
    else:
        tokenized_datasets = noisy_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", sentence1_key, sentence2_key],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")  
        
    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], 
                                 shuffle=False, 
                                 collate_fn=collate_fn, 
                                 batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], 
                                 shuffle=False, 
                                 collate_fn=collate_fn, 
                                 batch_size=batch_size)
    
    return train_dataloader, eval_dataloader,test_dataloader, noise_index, tokenized_datasets, collate_fn


def create_dataloaders_ER(model_name_or_path="roberta-large",
                       task="mrpc",
                       noise_ratio=0.1,
                       batch_size=32,
                       train_file="train.json",
                       valid_file="valid.json",
                       test_file="test.json",
                       max_length = 128,
                       select_index = None):
    ## train/valid/test file are pandas dataframe .json with 3 columns, sent_1, sent_2 and label. 0 means mismatch and 1 means match
    ## Default max_length is set to 128. For more complex task, set it to 256.
    
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Define task-specific keys for sentence1 and sentence2
    sentence1_key, sentence2_key = task_to_keys[task]
    
    def tokenize_function(examples, max_length=max_length):
        # Tokenize based on the task, considering whether there is a second sentence
        if sentence2_key is None:
            outputs = tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
        else:
            outputs = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, max_length=max_length)
        return outputs
    
    # Load the dataset from JSON files using pandas
    if train_file.__contains__('json'):
        train_df = pd.read_json(train_file)
        valid_df = pd.read_json(valid_file)
        test_df = pd.read_json(test_file)
    elif train_file.__contains__('csv'):
        train_df = pd.read_csv(train_file,index_col=0).reset_index(drop=True)
        valid_df = pd.read_csv(valid_file,index_col=0).reset_index(drop=True)
        test_df = pd.read_csv(test_file,index_col=0).reset_index(drop=True)     
    print(train_df.columns,valid_df.columns,test_df.columns)
    train_df.columns = ['sentence_1', 'sentence_2', 'label']
    valid_df.columns = ['sentence_1', 'sentence_2', 'label']
    test_df.columns = ['sentence_1', 'sentence_2', 'label']
    
    # Convert the DataFrames into the HuggingFace Dataset format
    if isinstance(select_index,np.ndarray):
        train_df = train_df.iloc[select_index].reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # If noise is required, apply the noise to the training set
    if noise_ratio > 0.0:
        n_train = len(train_dataset)
        noise_index = np.random.choice(n_train, size=int(noise_ratio * n_train), replace=False)
        def flip_label(example, ind, noise_index):
            if ind in noise_index:
                example["label"] = 1 - example["label"]
            return example
        train_dataset = train_dataset.map(flip_label, with_indices=True, fn_kwargs={'noise_index': noise_index})
    else:
        noise_index = []

    # Tokenize the dataset
    if sentence2_key is None: ## This is for single-sentence classification
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence_1"]  # Only remove sentence_1
        )
        tokenized_valid = valid_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence_1"]
        )
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence_1"]
        )
    else:
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence_1", "sentence_2"]
        )
        tokenized_valid = valid_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence_1", "sentence_2"]
        )
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["sentence_1", "sentence_2"]
        )

    # Rename the 'label' column to 'labels' for compatibility with models
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_valid = tokenized_valid.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")
    
    # Define the collate function for padding
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Create the DataLoader for train, validation, and test datasets
    tokenized_dataset = {}
    tokenized_dataset['train'] = tokenized_train
    tokenized_dataset['validation'] = tokenized_valid
    tokenized_dataset['test'] = tokenized_test
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(tokenized_dataset['validation'], 
                                 shuffle=False, 
                                 collate_fn=collate_fn, 
                                 batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_dataset['test'], 
                                 shuffle=False, 
                                 collate_fn=collate_fn, 
                                 batch_size=batch_size)
    
    return train_dataloader, eval_dataloader, test_dataloader, noise_index, tokenized_dataset, collate_fn