import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
task_to_keys = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "ER": ("sentence_1", "sentence_2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
}
def create_dataloaders(model_name_or_path="roberta-large",
                       task="mrpc",
                       noise_ratio=0.1,
                       batch_size=32,
                       train_file="train.json",
                       valid_file="valid.json",
                       test_file="test.json",
                       max_length = 128,
                       select_index = None):
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
    train_df = pd.read_json(train_file)
    valid_df = pd.read_json(valid_file)
    test_df = pd.read_json(test_file)
    train_df.columns = ['sentence_1', 'sentence_2', 'label']
    valid_df.columns = ['sentence_1', 'sentence_2', 'label']
    test_df.columns = ['sentence_1', 'sentence_2', 'label']
    
    if isinstance(select_index,np.ndarray):
        train_df = train_df.iloc[select_index].reset_index(drop=True)
    # Convert the DataFrames into the HuggingFace Dataset format
    
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
    if sentence2_key is None:
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
