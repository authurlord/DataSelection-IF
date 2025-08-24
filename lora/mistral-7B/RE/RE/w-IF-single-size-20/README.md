---
library_name: peft
license: other
base_model: /public/Mistral-7B-Instruct-v0.2
tags:
- base_model:adapter:/public/Mistral-7B-Instruct-v0.2
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: w-IF-single-size-20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# w-IF-single-size-20

This model is a fine-tuned version of [/public/Mistral-7B-Instruct-v0.2](https://huggingface.co//public/Mistral-7B-Instruct-v0.2) on the alpaca_en_demo dataset.

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
- num_devices: 2
- total_train_batch_size: 16
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.16.0
- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 2.16.1
- Tokenizers 0.21.0