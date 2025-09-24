---
library_name: peft
license: other
base_model: /home/wys/model/Mistral-7B-Instruct-v0.2
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: w-main-IF
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# w-main-IF

This model is a fine-tuned version of [/home/wys/model/Mistral-7B-Instruct-v0.2](https://huggingface.co//home/wys/model/Mistral-7B-Instruct-v0.2) on the alpaca_en_demo dataset.

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
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.49.0
- Pytorch 2.7.0+cu126
- Datasets 3.3.2
- Tokenizers 0.21.0