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

llm = LLM(model=output_path, 
            tensor_parallel_size=args.gpu_num,dtype="half", 
            enforce_eager=True,
            gpu_memory_utilization = args.gpu_memory_usage,
            enable_lora=True,
            disable_log_stats=True,)