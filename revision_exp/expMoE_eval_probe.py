# Exp B probe eval: probe-task F1 for a multi-task model (dense LoRA OR Poly+MHR).
#   --mode dense : base + standard LoRA adapter, served via vLLM (LoRARequest).
#   --mode poly  : base + PEFT Poly dir (n_splits>=1, MHR), DIRECT PEFT HF generation with task_ids
#                  (Poly+MHR cannot be baked to one standard LoRA, so we infer with PEFT directly).
# Probe tasks (fixed, evaluated at every n): ER-amazon-google, DI-amazon, SM-CMS.
# F1 via the PROVEN src/evaluation.py path (same shaping/extraction as poly_eval_metrics.py).
import argparse, os, json, io, ast, re, sys
from contextlib import redirect_stdout
import pandas as pd, numpy as np
sys.path.insert(0, os.getcwd())
from src.evaluation import evaluation

PROBE = [
    ('ER', 'amazon-google', 'train/ER/amazon-google/amazon-google-test.json', 'ER-amazon-google'),
    ('DI', 'amazon',        'train/DI/amazon/amazon-test.json',               'DI-amazon'),
    ('SM', 'CMS',           'train/SM/CMS/CMS-test.json',                     'SM-CMS'),
]

def parseable(p):
    try:
        v = ast.literal_eval(str(p).strip()); return isinstance(v, dict) and len(v) >= 1
    except Exception:
        return False

def sanitize(p):
    return p if parseable(p) else '{"_pred_": ""}'

def scalar_metric(task, m):
    if not isinstance(m, dict): return np.nan
    for k in ('F1', 'f1', 'micro_f1', 'Accuracy', 'Accuray', 'accuracy', 'acc', 'macro_f1'):
        if k in m and isinstance(m[k], (int, float)): return float(m[k])
    return np.nan

def rescue(raw):
    # Faithful format rescue (like normalize_poly_preds.py, extended): Poly can emit the correct VALUE
    # but a truncated dict (e.g. "'mismatch'}" / "'blank media'}") because poly_train masks the
    # prompt/response boundary tokens. Every handler reads dict .values()[0] (or a match/mismatch
    # substring), so we only need to recover the VALUE and wrap it in a valid dict. Never alters a row
    # that already parses; only the model's own predicted value is recovered (content unchanged).
    s = str(raw).strip()
    if parseable(s):
        return s
    s2 = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    if parseable(s2):
        return s2
    body = s2.strip().lstrip('{').rstrip('}').strip()
    val = body.split(':', 1)[1] if ':' in body else body      # drop any key part
    val = val.strip().strip('\'"').strip().replace('"', '').replace("'", '')
    return '{"v": "%s"}' % val if val else '{"_pred_": ""}'

def er_sm_safe(task, pred, gold):
    # Transfer() scores any ER/SM pred lacking 'mismatch'/'dismatch' as a POSITIVE match. A pred is only
    # trustworthy if it clearly states match OR mismatch; anything else (unparseable OR a parseable but
    # invalid dict) is forced to the OPPOSITE of gold (always wrong).
    if task in ('ER', 'SM'):
        p = str(pred).lower()
        said_mis = ('mismatch' in p) or ('dismatch' in p)
        said_match = ('match' in p) and not said_mis
        if said_mis or said_match:
            return pred
        g = str(gold).lower()
        return '{"Output": "match"}' if (('mismatch' in g) or ('dismatch' in g)) else '{"Output": "mismatch"}'
    if parseable(pred):
        return pred
    return '{"_pred_": ""}'

def score(task, ds, test, preds):
    ev = evaluation()
    golds = [str(t['output']) for t in test]
    preds = [er_sm_safe(task, rescue(p), gd) for p, gd in zip(preds, golds)]
    base = pd.DataFrame({'instruction': [t['instruction'] for t in test],
                         'input': [t.get('input', '') for t in test],
                         'output': golds, 'predict': preds})
    if task in ('ER', 'SM'):
        df = base[['instruction', 'input', 'output', 'predict']].copy()
    else:
        df = base[['output', 'predict']].copy()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            m = ev.process(task, ds, df)
        if not isinstance(m, dict):
            mm = re.search(r'Accuracy:([0-9.]+)', buf.getvalue())
            m = {'Accuracy': float(mm.group(1))} if mm else {}
    except Exception as e:
        print('  score err %s/%s: %s' % (task, ds, str(e)[:120])); m = {}
    return scalar_metric(task, m)

def make_prompt(d):
    return "[INST] %s [/INST]" % (d['instruction'] + (('\n' + d['input']) if d.get('input') else ''))

def run_dense(a):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.base, dtype='half', enable_lora=True, max_lora_rank=a.max_rank, max_loras=1,
              max_model_len=a.max_len, enforce_eager=True, gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(temperature=0, top_p=1, max_tokens=a.max_new)
    res = {}
    for task, ds, tf, name in PROBE:
        test = json.load(open(tf))[:a.cap]
        outs = llm.generate([make_prompt(d) for d in test], sp,
                            lora_request=LoRARequest('a', 1, a.adapter))
        preds = [o.outputs[0].text.strip() for o in outs]
        res[name] = score(task, ds, test, preds)
        print('%-18s F1=%.4f' % (name, res[name]))
    return res

def run_meld(a):
    # MELD reference line: each probe task routed to ITS OWN per-task expert (modular), scored on the
    # SAME probe rows/cap/metric as dense & poly -> a sound MELD baseline (not the Exp-A oracle shortcut).
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    EXPERT = {'ER-amazon-google': 'lora/mistral-7B/ER/amazon-google/select',
              'DI-amazon': 'lora/mistral-7B/DI/amazon/select',
              'SM-CMS': 'lora/mistral-7B/SM/CMS/select'}
    llm = LLM(model=a.base, dtype='half', enable_lora=True, max_lora_rank=a.max_rank, max_loras=1,
              max_model_len=a.max_len, enforce_eager=True, gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(temperature=0, top_p=1, max_tokens=a.max_new)
    res = {}
    for i, (task, ds, tf, name) in enumerate(PROBE):
        test = json.load(open(tf))[:a.cap]
        outs = llm.generate([make_prompt(d) for d in test], sp,
                            lora_request=LoRARequest(name, i + 1, EXPERT[name]))
        preds = [o.outputs[0].text.strip() for o in outs]
        res[name] = score(task, ds, test, preds)
        print('%-18s F1=%.4f (MELD expert)' % (name, res[name]))
    return res

def run_poly(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
                                                attn_implementation='eager').cuda().eval()
    model = PeftModel.from_pretrained(base, a.poly).eval()
    task2id = json.load(open(os.path.join(a.poly, 'task2id.json')))
    res = {}
    for task, ds, tf, name in PROBE:
        if name not in task2id:
            print('%-18s SKIP (not in this n: %s)' % (name, list(task2id))); res[name] = np.nan; continue
        tid = task2id[name]
        test = json.load(open(tf))[:a.cap]
        preds = []
        B = a.bsz
        for i in range(0, len(test), B):
            chunk = test[i:i + B]
            enc = tok([make_prompt(d) for d in chunk], return_tensors='pt', padding=True,
                      truncation=True, max_length=a.max_len - a.max_new).to('cuda')
            ti = torch.full((len(chunk),), tid, dtype=torch.long, device='cuda')
            with torch.no_grad():
                out = model.generate(**enc, task_ids=ti, max_new_tokens=a.max_new,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc['input_ids'].shape[1]:]
            preds += [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        res[name] = score(task, ds, test, preds)
        print('%-18s F1=%.4f (task_id=%d)' % (name, res[name], tid))
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['dense', 'poly', 'meld'])
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter')      # dense LoRA dir
    ap.add_argument('--poly')         # PEFT Poly dir
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='revision_exp/results/expB_probe/probe_f1.csv')
    ap.add_argument('--cap', type=int, default=6000)  # full probe test (SM-CMS=5127, 0.7% positive -> need all)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--gpu_mem', type=float, default=0.85)
    ap.add_argument('--max_len', type=int, default=1536)
    ap.add_argument('--max_new', type=int, default=64)
    ap.add_argument('--max_rank', type=int, default=16)
    a = ap.parse_args()
    if a.mode == 'dense':
        assert a.adapter, '--adapter required for --mode dense'
    elif a.mode == 'poly':
        assert a.poly, '--poly required for --mode poly'
    res = {'dense': run_dense, 'poly': run_poly, 'meld': run_meld}[a.mode](a)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    row = {'tag': a.tag, 'mode': a.mode, **{k: round(float(v), 4) for k, v in res.items()},
           'probe_mean': round(float(np.nanmean(list(res.values()))), 4)}
    hdr = not os.path.exists(a.out)
    pd.DataFrame([row]).to_csv(a.out, mode='a', header=hdr, index=False)
    print('APPENDED', a.out, json.dumps(row))
