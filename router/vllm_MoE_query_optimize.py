import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sklearn.metrics import f1_score
import os
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml
import argparse
import json
import subprocess

model_path = '/data/home/wangys/model/Mistral-7B-Instruct-v0.2'
file_path = '../train/router/test.csv'
# expert_index_path = '../train/router/expert_index.npy'
# expert_index = np.load(expert_index_path,allow_pickle=True).item()

llm = LLM(model=model_path, 
            tensor_parallel_size=8, 
            enforce_eager=True,
            gpu_memory_utilization = 0.7,
            enable_lora=True,
            # enable_chunked_prefill=True,
            disable_log_stats=True,
            # num_scheduler_steps=10,
            max_lora_rank=64
            )

tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(temperature=0, 
                                    top_p=1,
                                    max_tokens=64) ## Maximum token size at 42

result = pd.read_csv(file_path,index_col=0)
result['prediction'] = ''

selection_list = list(result['output'].unique())
selection_list = [s for s in selection_list if s!='']
expert_combination = list(result['labels'].value_counts().index.tolist())

expert_combination_dict = {}

expert_count = 0
for index in range(len(expert_combination)): ## extract standalone loras
    expert_0,expert_1 = eval(expert_combination[index].replace('/','--'))
    # lora_path = expert_index[expert_name]['lora_path']
    expert_name = '{}|{}'.format(expert_0,expert_1)
    lora_path = '/data/home/wangys/Expert_Combination/{}'.format(expert_name)
    expert_combination_dict[expert_name] = expert_count
    expert_count += 1
# expert_conbination = [e.replace('/','--') for e in expert_conbination] ## 格式对齐


task_list = list(result['task'].unique())


print(len(selection_list))


# text_list = result['instruction'].to_list()

# if(model_path.lower().__contains__('mistral')):
#     text_all = ["[INST] %s [/INST]" % str(a) for a in text_list]
# else: 
#     text_all = []
#     for prompt in tqdm(text_list):
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ]
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         text_all.append(text)
for task in task_list:
    selection_list = list(result[result['task']==task]['output'].unique())
    selection_list = [s for s in selection_list if s!='']
    print('Selection Number {} for task: {}'.format(len(selection_list),task))

    for index in range(len(expert_combination)): ## extract standalone loras
        expert_0,expert_1 = eval(expert_combination[index].replace('/','--'))
        # lora_path = expert_index[expert_name]['lora_path']
        expert_name = '{}|{}'.format(expert_0,expert_1)
        lora_path = '/data/home/wangys/Expert_Combination/{}'.format(expert_name)
        
        expert_combination = list(result[result['task']==task]['labels'].unique())
        select_index = result[(result['labels']==expert_combination[index]) & (result['task']==task)].index
        print(len(select_index),expert_combination[index],task)

    ## Select File
        result_df = result.iloc[select_index]


        text_list = result.iloc[select_index,0].to_list()

        if(model_path.lower().__contains__('mistral')):
            text_all = ["[INST] %s [/INST]" % str(a) for a in text_list]
        else: 
            text_all = []
            for prompt in tqdm(text_list):
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                text_all.append(text)
        
        
        # outputs = llm.generate(text_all, sampling_params,lora_request = LoRARequest(expert_name, index, lora_path))  
        # outputs = llm.generate(text_all, sampling_params,lora_request = LoRARequest(expert_name, index, lora_path),guided_options_request=dict(guided_choice=selection_list)) 
        outputs = llm.generate(text_all, sampling_params,lora_request = LoRARequest(expert_name, expert_combination_dict[expert_name], lora_path),
                            guided_options_request=dict(guided_choice=selection_list)
                            ) 
        ## retrieve output to list
        generation_list = []
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generation_list.append(generated_text)
        ## write text
        # result[expert_name] = generation_list
        # result.to_csv('../train/router/train_output_guided.csv')
        count = 0
        for select_index_single in select_index:
            result.iloc[select_index_single,-1] = generation_list[count]
            count += 1
        print(count,len(select_index))
        result.to_csv('../train/router/test_guided_MoE_optimized.csv')
result.to_csv('../train/router/test_guided_MoE_optimized.csv')
        

        