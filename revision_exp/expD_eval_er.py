# P0-1 helper: eval an ER/amazon-google LoRA adapter -> F1 (vLLM + src.evaluation handle_ER).
import os, sys, json, argparse, ast, io, re
from contextlib import redirect_stdout
import pandas as pd
sys.path.insert(0, os.getcwd())
from src.evaluation import evaluation

def parseable(p):
    try:
        v = ast.literal_eval(str(p).strip()); return isinstance(v, dict) and len(v) >= 1
    except Exception:
        return False

def er_safe(pred, gold):
    p = str(pred).lower()
    if ('mismatch' in p) or ('dismatch' in p) or (('match' in p) and 'mismatch' not in p):
        return pred
    g = str(gold).lower()
    return '{"Output": "match"}' if (('mismatch' in g) or ('dismatch' in g)) else '{"Output": "mismatch"}'

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True); ap.add_argument('--adapter', required=True)
ap.add_argument('--test', default='train/ER/amazon-google/amazon-google-test.json')
ap.add_argument('--tag', required=True); ap.add_argument('--out', default='revision_exp/results/expD_cohesion_f1_raw.csv')
ap.add_argument('--cap', type=int, default=2400); ap.add_argument('--max_new', type=int, default=32)
a = ap.parse_args()
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
test = json.load(open(a.test))[:a.cap]
llm = LLM(model=a.base, dtype='half', enable_lora=True, max_lora_rank=16, max_loras=1,
          max_model_len=1536, enforce_eager=True, gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=0, top_p=1, max_tokens=a.max_new)
prompts = ['[INST] %s [/INST]' % (d['instruction'] + (('\n' + d['input']) if d.get('input') else '')) for d in test]
outs = llm.generate(prompts, sp, lora_request=LoRARequest('a', 1, a.adapter))
golds = [str(d['output']) for d in test]
preds = [er_safe(o.outputs[0].text.strip(), g) for o, g in zip(outs, golds)]
df = pd.DataFrame({'instruction': [d['instruction'] for d in test], 'input': [d.get('input', '') for d in test],
                   'output': golds, 'predict': preds})
buf = io.StringIO()
with redirect_stdout(buf):
    m = evaluation().process('ER', 'amazon-google', df)
f1 = float(m.get('F1', m.get('f1', float('nan'))))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
hdr = not os.path.exists(a.out)
pd.DataFrame([{'tag': a.tag, 'f1': round(f1, 4)}]).to_csv(a.out, mode='a', header=hdr, index=False)
print('%s  ER-amazon-google F1=%.4f' % (a.tag, f1))
