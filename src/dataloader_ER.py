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

## Batch Approximation:在read_dataframe中需要加上id这一个key
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
    train_df['id'] = train_df.index
    valid_df['id'] = valid_df.index
    test_df['id'] = test_df.index
    train_df.columns = ['sentence_1', 'sentence_2', 'labels','id']
    valid_df.columns = ['sentence_1', 'sentence_2', 'labels','id']
    test_df.columns = ['sentence_1', 'sentence_2', 'labels','id']
    
    if isinstance(select_index,np.ndarray):
        train_df = train_df.iloc[select_index].reset_index(drop=True)
    # Convert the DataFrames into the HuggingFace Dataset format
    
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)
    

    # If noise is required, apply the noise to the training set
    # if noise_ratio > 0.0:
    #     n_train = len(train_dataset)
    #     noise_index = np.random.choice(n_train, size=int(noise_ratio * n_train), replace=False)
    #     def flip_label(example, ind, noise_index):
    #         if ind in noise_index:
    #             example["label"] = 1 - example["label"]
    #         return example
    #     train_dataset = train_dataset.map(flip_label, with_indices=True, fn_kwargs={'noise_index': noise_index})
    # else:
    #     noise_index = []

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
    # tokenized_train = tokenized_train.rename_column("label", "labels")
    # tokenized_valid = tokenized_valid.rename_column("label", "labels")
    # tokenized_test = tokenized_test.rename_column("label", "labels")
    
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
    
    return train_dataloader, eval_dataloader, test_dataloader, tokenized_dataset, collate_fn



def create_dataloaders_WebTable(
        model_name_or_path="roberta-large",
        task="mrpc",
        batch_size=32,
        train_file="train.csv",
        valid_file="valid.csv",
        test_file="test.csv",
        max_length=128,
        select_index=None
    ):
    """
    Create dataloaders for training, validation, and testing with label mapping.

    Args:
        model_name_or_path (str): Path to the pretrained model or model name.
        task (str): Task name, e.g., "mrpc".
        batch_size (int): Batch size for dataloaders.
        train_file (str): Path to the training CSV file.
        valid_file (str): Path to the validation CSV file.
        test_file (str): Path to the testing CSV file.
        max_length (int): Maximum sequence length.
        select_index (np.ndarray): Optional subset indices for the training data.

    Returns:
        train_dataloader, eval_dataloader, test_dataloader, tokenized_dataset, collate_fn, label_to_int, int_to_label
    """
    import pandas as pd
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from datasets import Dataset
    import numpy as np

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the datasets from CSV files
    train_df = pd.read_csv(train_file, index_col=0)
    valid_df = pd.read_csv(valid_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)

    # Ensure the dataframes have the correct columns
    train_df.columns = ['sentence_1', 'sentence_2', 'labels', 'index_all']
    valid_df.columns = ['sentence_1', 'sentence_2', 'labels', 'index_all']
    test_df.columns = ['sentence_1', 'sentence_2', 'labels', 'index_all']
    
    train_df['id'] = train_df.index
    valid_df['id'] = valid_df.index
    test_df['id'] = test_df.index

    # Combine all labels from train, valid, and test to create a mapping
    all_labels = pd.concat([train_df['labels'], valid_df['labels'], test_df['labels']]).unique()
    label_to_int = {label: idx for idx, label in enumerate(all_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    # Map the labels to integers
    train_df['labels'] = train_df['labels'].map(label_to_int)
    valid_df['labels'] = valid_df['labels'].map(label_to_int)
    test_df['labels'] = test_df['labels'].map(label_to_int)

    # Convert the DataFrames into HuggingFace Dataset format
    if isinstance(select_index, np.ndarray):
        train_df = train_df.iloc[select_index].reset_index(drop=True)
    
    # train_dataset = Dataset.from_pandas(train_df)
    # valid_dataset = Dataset.from_pandas(valid_df)
    # test_dataset = Dataset.from_pandas(test_df)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_dataset = Dataset.from_pandas(valid_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Define tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence_1"], 
            examples["sentence_2"], 
            truncation=True, 
            max_length=max_length,
        )

    # Tokenize datasets
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["sentence_1", "sentence_2", "index_all"]
    )
    tokenized_valid = valid_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["sentence_1", "sentence_2", "index_all"]
    )
    tokenized_test = test_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["sentence_1", "sentence_2", "index_all"]
    )

    # Define the collate function for padding
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    tokenized_datasets = {}
    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_train, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_valid, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    test_dataloader = DataLoader(
        tokenized_test, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    tokenized_datasets['train'] = tokenized_train
    tokenized_datasets['validation'] = tokenized_valid
    tokenized_datasets['test'] = tokenized_test
    return train_dataloader, eval_dataloader, test_dataloader, tokenized_datasets, collate_fn, len(label_to_int), int_to_label
