---
base_model: /data/home/wangys/model/Mistral-7B-Instruct-v0.2
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: select-1k
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# select-1k

This model is a fine-tuned version of [/data/home/wangys/model/Mistral-7B-Instruct-v0.2](https://huggingface.co//data/home/wangys/model/Mistral-7B-Instruct-v0.2) on the Abt-Buy-Match-P1 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 3
- total_train_batch_size: 24
- total_eval_batch_size: 24
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.4.0+cu121
- Datasets 2.20.0
- Tokenizers 0.20.3