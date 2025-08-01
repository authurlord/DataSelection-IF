import subprocess
from time import sleep
from tqdm import tqdm
import time

command_line = []
task_list = [
    ['RE','RE'],
    ['ER','abt-buy'],
    ['ER','walmart-amazon'],
    ['ER','amazon-google'],
    ['ER','wdc'],
    ['ER','semi-text-w'],
    ['DC','hospital'],
    ['DC','beer'],
    ['DC','rayyan'],
    ['DI','amazon'],
    ['DI','walmart'],
    ['AVE','oa_mine'],
    ['CTA','SimTab'],
    ['CTA','WebTable']
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
    #     'python QuRating_Selection.py --task {} --dataset {} --device 6 --model_path ../../model/QuRating-1.3B'.format(task,dataset)
    # )
    command_line.append(
        'python QuRating_Selection_top_100.py --task {} --dataset {} --device 6 --model_path ../../model/QuRating-1.3B'.format(task,dataset)
    )
    ### Test IF with DataInf
    # command_line.append(
    #     'python Ablation_Study.py --yaml_path script/config_{}_{}.yaml --device 6 --model mistral-7B --DO_TRAIN_IF_SINGLE --DO_EVAL_IF_SINGLE'.format(task,dataset)
    # )
    
for command in command_line:
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    print("标准输出：", result.stdout)


    
