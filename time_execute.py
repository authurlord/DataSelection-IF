import subprocess
from time import sleep
from tqdm import tqdm
import time

command_line = []
# task_list = [
#     ['ER','semi-text-c'], ## from here
#     ['DC','hospital'],
#     ['DC','rayyan'],
#     ['DI','amazon'],
#     ['DI','walmart'],
#     ['CTA','WebTable'],
#     ['CTA','SimTab'],
# ]
task_list = [
    ['DI','amazon'],
    ['DI','walmart'],
    ['CTA','WebTable'],
    ['CTA','SimTab'],
]
for task,dataset in task_list:
    # if task=='CTA':
    #     command_line.append(
    #         'python Ablation_Study.py --yaml_path script/config_{}_{}.yaml --device 4,5 --model mistral-7B --DO_TRAIN --DO_EVAL'.format(task,dataset)
    #     ) ## 不训练Main模型，已经有结果了！
    # else:
    #     command_line.append(
    #         'python Ablation_Study.py --yaml_path script/config_{}_{}.yaml --device 4,5 --model mistral-7B --DO_TRAIN --DO_EVAL --DO_TRAIN_MAIN --DO_EVAL_MAIN'.format(task,dataset)
    #     )
    ### IF with small model
    # command_line.append(
    #     'CUDA_VISIBLE_DEVICES=6 python cal_IF_pipeline.py --yaml_path script/config_{}_{}.yaml --IF_device cuda'.format(task,dataset)
    # )
    command_line.append(
        'python Ablation_Study.py --yaml_path script/config_{}_{}.yaml --device 6 --model mistral-7B --DO_TRAIN_QURATING --DO_EVAL_QURATING --DO_SELECT'.format(task,dataset)
    )
    ### Test IF with DataInf
    # command_line.append(
    #     'python Ablation_Study.py --yaml_path script/config_{}_{}.yaml --device 6 --model mistral-7B --DO_TRAIN_IF_SINGLE --DO_EVAL_IF_SINGLE'.format(task,dataset)
    # )
    
for command in command_line:
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    print("标准输出：", result.stdout)


    
