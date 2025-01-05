from transformers import AutoTokenizer
from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig, GenerationConfig
import numpy as np
import pandas as pd
from tqdm import tqdm
# load model and tokenizer
model_repoid_or_path = '/data/home/wangys/model/Qwen2.5-0.5B-Instruct'
backend_config = TurbomindEngineConfig(
        rope_scaling_factor=2.5,
        session_len=1000000,
        cache_max_entry_count=0.7,
        tp=2,
        max_prefill_iters=10)

gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=256)

pipe = pipeline(model_repoid_or_path, 
                backend_config=backend_config,cha)
tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path, trust_remote_code=True)

text_list = pd.read_json('/data/home/wangys/DataSelection-IF/train/RE/RE-train.json').iloc[:,0].to_list()

text_all = []
for prompt in tqdm(text_list):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text_all.append(messages)
text_all = tokenizer.apply_chat_template(
    text_all
)

# input_ids = tokenizer.apply_chat_template(messages)
# logits = pipe.get_logits(input_ids)

# ppl
ppl = pipe.get_ppl(text_all)

np.save('RE_ppl.npy',ppl)
print(ppl)

# messages = [
#    {"role": "user", "content": "Hello, how are you?"},
# ]
# input_ids = tokenizer.apply_chat_template(messages)
# # logits = pipe.get_logits(input_ids)

# # ppl
# ppl = pipe.get_ppl(input_ids)
# print(ppl)