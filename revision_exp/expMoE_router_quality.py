# Unified router experiment (R1-2 + R1-4 + AE-5). ONE sweep over expert count n yields:
#   (1) SCALABILITY : routing -> real task F1 vs oracle (in-domain expert) vs random, as n grows.
#   (2) STABILITY   : expert utilization balance + decision entropy + ACROSS-SEED variance, as n grows.
#   (3) COMPLEXITY  : router fit wall-clock + trainable params, as n grows (router does NOT retrain experts).
#
# Router is trained on QUERY CONTENT (input field, template stripped), quality measured as TASK F1 after routing.
#
# CORRECTNESS FIXES over the v2 draft (verified against src/evaluation.py + poly_eval_metrics.py):
#   * Subsample is FIRST-N IN ORDER (not random) -> preserves positional alignment for RE's handle_RE_RE
#     (it reads gold from train/RE/test_RAG.csv by row position; a random subset would misalign gold).
#   * Per-task metric extraction ported from the PROVEN poly_eval_metrics.py:
#       - per-task df shaping (ER/SM: 4 cols; CTA: out/pred/index; DI/AVE/RE/DC: out/pred)
#       - DI's handle_imputation has NO return (only prints "Accuracy:") -> capture from stdout
#       - AVE key is misspelled 'Accuray'; RE returns 'micro_f1'/'acc'; ER/SM/DC return 'F1'
#       - unparseable predictions sanitized to '{"_pred_": ""}' (count as wrong, never crash a handler)
#
# Two phases (run --phase predict ONCE on GPU, then --phase route on CPU repeatedly):
#   conda activate deepspeed   # 51.11, vllm 0.5.4
#   python expMoE_router_quality.py --phase predict --manifest manifest.csv \
#       --base /data/yanmengyi/huggingface/Mistral-7B-Instruct-v0.2 --per_ds 300 \
#       --cache revision_exp/results/router_quality/predmatrix
#   python expMoE_router_quality.py --phase route --manifest manifest.csv \
#       --bge /data/yanmengyi/sentence_transformer_model/bge-large-en-1.5 \
#       --cache revision_exp/results/router_quality/predmatrix \
#       --ns 3,4,5,6,7,8,9,10,11,12 --seeds 0,1,2 --out revision_exp/results/router_quality

import argparse, os, json, time, glob, io, ast, re
from contextlib import redirect_stdout
import numpy as np, pandas as pd

# ----------------------------- shared helpers -----------------------------
def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def strip_template(instruction, inp):
    # Route on the INSTANCE CONTENT, not the task template (else routing degenerates to the trivial
    # template->dataset classification the revision guide explicitly rejects). Task families lay the
    # instance out differently (ER/SM/DI: instance AFTER the format example; RE: Table block BEFORE it;
    # AVE: target after few-shot), so instead of one split rule we REMOVE the boilerplate phrases that
    # are identical across datasets of a family, wherever they appear, keeping the instance records.
    inp = (inp or "").strip()
    if inp:
        return inp
    s = str(instruction)
    s = re.sub(r'(?is)you are .*?(?:json format\.?|judgement\.?\s|judgement\.?$)', ' ', s, count=1)
    s = re.sub(r'(?is)return in json format\.?', ' ', s)
    s = re.sub(r'(?is)output format example\s*:?\s*\{[^}]*\}', ' ', s)
    s = re.sub(r'(?is)(?:relation\s+)?options?\s*:\s*\[[^\]]*\]', ' ', s)
    # drop residual generic task sentences (shared across all datasets of a family) -> route on instance
    s = re.sub(r'(?is)judge whether .*?(?:options|mismatch)\.?', ' ', s)
    s = re.sub(r'(?is)required attribute for extraction is .*?(?:n/?a\'?\.|\.)', ' ', s)
    s = re.sub(r'(?is)given (?:information from|table title and column pair)[^.]*\.', ' ', s)
    for cut in ['take these examples', 'take these rows', 'for example']:
        i = s.lower().find(cut)
        if i >= 0:
            s = s[:i]                                        # drop few-shot demonstration block
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:1500] if len(s) >= 20 else str(instruction)[-400:]

def load_manifest(path):
    m = pd.read_csv(path)
    assert set(['task', 'dataset', 'adapter', 'test_file']).issubset(m.columns)
    return m.reset_index(drop=True)

def parseable(p):
    try:
        v = ast.literal_eval(str(p).strip()); return isinstance(v, dict) and len(v) >= 1
    except Exception:
        return False

def sanitize(p):
    return p if parseable(p) else '{"_pred_": ""}'

def rescue(raw):
    # Faithful format rescue: recover the predicted VALUE from a truncated dict ("'x'}" / bare value)
    # and rewrap as a valid dict (handlers read .values()[0]). Never alters an already-parseable row.
    s = str(raw).strip()
    if parseable(s):
        return s
    s2 = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    if parseable(s2):
        return s2
    body = s2.strip().lstrip('{').rstrip('}').strip()
    val = body.split(':', 1)[1] if ':' in body else body
    val = val.strip().strip('\'"').strip().replace('"', '').replace("'", '')
    return '{"v": "%s"}' % val if val else '{"_pred_": ""}'

def er_sm_safe(task, pred, gold):
    # ER/SM: Transfer() scores any pred lacking 'mismatch'/'dismatch' as a POSITIVE match. So a pred is
    # only trustworthy if it clearly states match OR mismatch; anything else (unparseable OR a parseable
    # but invalid dict like {"foo":"bar"}) must be forced to the OPPOSITE of gold (always wrong).
    if task in ('ER', 'SM'):
        p = str(pred).lower()
        said_mis = ('mismatch' in p) or ('dismatch' in p)
        said_match = ('match' in p) and not said_mis
        if said_mis or said_match:
            return pred                                     # recognizable label -> let Transfer score it
        g = str(gold).lower()
        gold_neg = ('mismatch' in g) or ('dismatch' in g)
        return '{"Output": "match"}' if gold_neg else '{"Output": "mismatch"}'
    if parseable(pred):
        return pred
    return '{"_pred_": ""}'   # DI/AVE/RE handlers already score this as wrong

# scalar F1/accuracy extraction per task, matching src/evaluation.py return keys
def scalar_metric(task, m):
    if not isinstance(m, dict):
        return np.nan
    for k in ('F1', 'f1', 'micro_f1', 'Accuracy', 'Accuray', 'accuracy', 'acc', 'macro_f1'):
        if k in m and isinstance(m[k], (int, float)):
            return float(m[k])
    return np.nan

# ----------------------------- phase 1: prediction matrix -----------------------------
def phase_predict(a):
    import sys; sys.path.insert(0, os.getcwd())
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    man = load_manifest(a.manifest)
    os.makedirs(a.cache, exist_ok=True)

    qpath = os.path.join(a.cache, 'queries.parquet')
    if os.path.exists(qpath):
        Q = pd.read_parquet(qpath)
        print('loaded query set:', len(Q))
    else:
        rows = []
        for _, r in man.iterrows():
            data = json.load(open(r['test_file']))
            if a.per_ds and len(data) > a.per_ds:
                data = data[:a.per_ds]                       # FIRST-N IN ORDER (positional-safe for RE)
            for d in data:
                rows.append({'dataset': r['dataset'], 'task': r['task'],
                             'instruction': d['instruction'], 'input': d.get('input', ''),
                             'gold': str(d['output'])})
        Q = pd.DataFrame(rows)
        Q['qid'] = np.arange(len(Q))
        Q.to_parquet(qpath)
        print('built query set: %d queries over %d datasets -> %s' % (len(Q), Q.dataset.nunique(), qpath))

    tok = AutoTokenizer.from_pretrained(a.base)
    llm = LLM(model=a.base, dtype='half', enable_lora=True, max_lora_rank=a.max_rank, max_loras=2,
              max_model_len=a.max_len, enforce_eager=True, gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(temperature=0, top_p=1, max_tokens=a.max_new)
    budget = a.max_len - a.max_new - 8

    prompts = []
    for _, q in Q.iterrows():
        s = "[INST] %s [/INST]" % (q['instruction'] + (('\n' + q['input']) if q['input'] else ''))
        ids = tok(s, add_special_tokens=True)['input_ids'][:budget]
        prompts.append({'prompt_token_ids': ids})

    for ei, r in man.iterrows():
        outp = os.path.join(a.cache, 'expert_%02d_%s.parquet' % (ei, r['dataset']))
        if os.path.exists(outp):
            print('skip (cached):', outp); continue
        if not os.path.exists(os.path.join(r['adapter'], 'adapter_config.json')) or \
           not os.path.exists(os.path.join(r['adapter'], 'adapter_model.safetensors')):
            print('expert %d (%s) MISSING ADAPTER %s -> skip' % (ei, r['dataset'], r['adapter'])); continue
        try:
            outs = llm.generate(prompts, sp, lora_request=LoRARequest(str(ei), ei + 1, r['adapter']))
            pred = [o.outputs[0].text.strip() for o in outs]
            pd.DataFrame({'qid': Q['qid'].values, 'predict': pred}).to_parquet(outp)
            print('expert %d (%s): predicted %d -> %s' % (ei, r['dataset'], len(pred), outp))
        except Exception as e:
            print('expert %d (%s) FAILED: %s' % (ei, r['dataset'], str(e)[:200]))
    print('PHASE_PREDICT_DONE')

# ----------------------------- phase 2: routing analysis -----------------------------
def phase_route(a):
    import sys; sys.path.insert(0, os.getcwd())
    from src.evaluation import evaluation
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    ev = evaluation()
    man = load_manifest(a.manifest)
    os.makedirs(a.out, exist_ok=True)

    Q = pd.read_parquet(os.path.join(a.cache, 'queries.parquet'))
    M = {}
    for ei in range(len(man)):
        f = glob.glob(os.path.join(a.cache, 'expert_%02d_*.parquet' % ei))
        if not f:
            print('WARN missing expert %d predictions; excluded from sweep' % ei); continue
        M[ei] = pd.read_parquet(f[0]).set_index('qid')['predict']
    have = sorted(M.keys())
    ds_of_expert = {ei: man.loc[ei, 'dataset'] for ei in have}
    task_of_ds = dict(zip(man['dataset'], man['task']))
    expert_of_ds = {man.loc[ei, 'dataset']: ei for ei in have}
    print('experts with predictions:', [ds_of_expert[e] for e in have])

    bge = SentenceTransformer(a.bge, device=a.bge_device)
    content = [strip_template(r['instruction'], r['input']) for _, r in Q.iterrows()]
    print('encoding %d query contents (template stripped)...' % len(content))
    Emb = bge.encode(content, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    Q = Q.assign(_row=np.arange(len(Q)))

    def eval_one_dataset(ds, g, chosen_experts):
        # g: rows of one dataset (sorted by qid -> positional alignment preserved for RE/CTA/DC).
        g = g.sort_values('qid')
        task = task_of_ds[ds]
        raw = [M[e].get(int(q), '') if e in M else '' for e, q in zip(chosen_experts, g['qid'].values)]
        golds = list(g['gold'].values)
        # rescue truncated/loose dict format, then force ER/SM invalid preds to the opposite of gold.
        preds = [er_sm_safe(task, rescue(p), gd) for p, gd in zip(raw, golds)]
        base = pd.DataFrame({'instruction': g['instruction'].values, 'input': g['input'].values,
                             'output': g['gold'].values, 'predict': preds})
        if task in ('ER', 'SM'):
            df = base[['instruction', 'input', 'output', 'predict']].copy()
        elif task == 'CTA':
            df = base[['output', 'predict']].copy(); df['index'] = np.arange(len(df))
        else:                                   # DI / AVE / RE / DC
            df = base[['output', 'predict']].copy()
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                m = ev.process(task, ds, df)
            if not isinstance(m, dict):
                mm = re.search(r'Accuracy:([0-9.]+)', buf.getvalue())
                m = {'Accuracy': float(mm.group(1))} if mm else {}
        except Exception as e:
            print('  eval err %s/%s: %s' % (task, ds, str(e)[:120])); m = {}
        return scalar_metric(task, m)

    def f1_for_subset(qsub, route_fn):
        f1s = []
        for ds, g in qsub.groupby('dataset'):
            g = g.sort_values('qid')
            chosen = [route_fn(i) for i in g['_row'].values]
            f1s.append(eval_one_dataset(ds, g, chosen))
        return float(np.nanmean(f1s))

    ns = [int(x) for x in a.ns.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    rows = []
    all_ds = [ds_of_expert[e] for e in have]
    for n in ns:
        if n > len(have):
            print('skip n=%d (only %d experts available)' % (n, len(have))); continue
        per_seed = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            keep_ds = list(rng.permutation(all_ds))[:n]
            keep_e = [expert_of_ds[d] for d in keep_ds]
            qsub = Q[Q['dataset'].isin(keep_ds)].copy()
            # NO train==test leakage: per dataset, router-EVAL = first half (contiguous from qid 0 so RE's
            # positional gold in test_RAG.csv stays aligned), router-TRAIN = second half (disjoint queries).
            tr_parts, te_parts = [], []
            for ds, gg in qsub.groupby('dataset'):
                gg = gg.sort_values('qid'); h = len(gg) // 2
                te_parts.append(gg.iloc[:h]); tr_parts.append(gg.iloc[h:])
            qtr = pd.concat(tr_parts); qte = pd.concat(te_parts)
            if n < 2:                                  # single expert: no classifier needed
                pred_ds = np.array([keep_ds[0]] * len(qte)); fit_s = 0.0; n_params = 0
                util = 1.0; dec_ent = 0.0
            else:
                Xtr, ytr = Emb[qtr['_row'].values], qtr['dataset'].values
                t0 = time.time()
                clf = LogisticRegression(max_iter=2000, C=10).fit(Xtr, ytr)
                fit_s = time.time() - t0
                n_params = int(clf.coef_.size + clf.intercept_.size)
                proba = clf.predict_proba(Emb[qte['_row'].values]); pred_ds = clf.classes_[proba.argmax(1)]
                _, cnt = np.unique(pred_ds, return_counts=True); load = cnt / cnt.sum()
                util = entropy(load) / np.log(n)
                dec_ent = float(np.mean([entropy(p) for p in proba]))
            te_idx = qte['_row'].values
            row2pos = {r: k for k, r in enumerate(te_idx)}
            def route_router(i):
                return expert_of_ds[pred_ds[row2pos[i]]]
            def route_oracle(i):
                return expert_of_ds[Q.loc[Q['_row'] == i, 'dataset'].values[0]]
            rstate = np.random.default_rng(1000 + sd)
            def route_random(i):
                return int(rstate.choice(keep_e))
            f1_router = f1_for_subset(qte, route_router)   # F1 only on held-out router-EVAL queries
            f1_oracle = f1_for_subset(qte, route_oracle)
            f1_random = f1_for_subset(qte, route_random)
            route_acc = float((pred_ds == qte['dataset'].values).mean())
            per_seed.append(dict(f1_router=f1_router, f1_oracle=f1_oracle, f1_random=f1_random,
                                 util=util, dec_ent=dec_ent, route_acc=route_acc,
                                 fit_s=fit_s, n_params=n_params))
        agg = {k: (round(float(np.mean([d[k] for d in per_seed])), 4),
                   round(float(np.std([d[k] for d in per_seed])), 4)) for k in per_seed[0]}
        row = {'n': n}
        for k, (mu, sd_) in agg.items():
            row[k + '_mean'] = mu; row[k + '_std'] = sd_
        rows.append(row)
        print('n=%2d  F1 router=%.3f±%.3f oracle=%.3f random=%.3f | util=%.3f±%.3f dec_ent=%.3f | acc=%.3f fit=%.2fs params=%d'
              % (n, row['f1_router_mean'], row['f1_router_std'], row['f1_oracle_mean'], row['f1_random_mean'],
                 row['util_mean'], row['util_std'], row['dec_ent_mean'], row['route_acc_mean'],
                 row['fit_s_mean'], row['n_params_mean']))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, 'router_quality_sweep.csv'), index=False)
    json.dump(rows, open(os.path.join(a.out, 'router_quality_sweep.json'), 'w'), indent=1)
    print('SAVED', os.path.join(a.out, 'router_quality_sweep.csv'))
    print('PHASE_ROUTE_DONE')

# ----------------------------- main -----------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['predict', 'route'])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--base'); ap.add_argument('--bge')
    ap.add_argument('--bge_device', default='cuda')
    ap.add_argument('--per_ds', type=int, default=300)
    ap.add_argument('--ns', default='3,4,5,6,7,8,9,10,11,12')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', default='revision_exp/results/router_quality')
    ap.add_argument('--gpu_mem', type=float, default=0.85)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--max_new', type=int, default=64)
    ap.add_argument('--max_rank', type=int, default=16)
    a = ap.parse_args()
    if a.phase == 'predict':
        assert a.base, '--base required for predict'
        phase_predict(a)
    else:
        assert a.bge, '--bge required for route'
        phase_route(a)
