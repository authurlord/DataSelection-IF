# P0-2 Old-task routing consistency (R1-4): adding experts changes the router's candidate pool.
# Does an OLD-task query still route to its original in-domain expert as the pool grows 3 -> 12?
# No retraining: reuse the predmatrix queries + bge content embeddings; refit the content router per
# nested task set and route the FIXED probe queries (ER-amazon-google, DI-amazon).
import os, sys, json, argparse, re
import numpy as np, pandas as pd

def strip_template(instruction, inp):
    inp = (inp or '').strip()
    if inp:
        return inp
    s = str(instruction)
    if 'Output format example' in s:
        s = s.split('Output format example', 1)[1]
        s = s.split('\n\n', 1)[1] if '\n\n' in s else s
    s = re.sub(r'(?is)judge whether .*?(?:options|mismatch)\.?', ' ', s)
    s = re.sub(r'(?is)(?:relation\s+)?options?\s*:\s*\[[^\]]*\]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()[:1500]

# nested task sets (probe always first 3); probe routing tracked = ER-amazon-google + DI-amazon
NESTED = {
    3:  ['amazon-google', 'amazon', 'CMS'],
    6:  ['amazon-google', 'amazon', 'CMS', 'walmart-amazon', 'wdc', 'RE'],
    9:  ['amazon-google', 'amazon', 'CMS', 'walmart-amazon', 'wdc', 'RE', 'abt-buy', 'semi-text-w', 'walmart'],
    12: ['amazon-google', 'amazon', 'CMS', 'walmart-amazon', 'wdc', 'RE', 'abt-buy', 'semi-text-w', 'walmart',
         'semi-text-c', 'synthea', 'oa_mine'],
}
PROBE = ['amazon-google', 'amazon']   # ER-amazon-google, DI-amazon

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default='revision_exp/results/router_quality/predmatrix')
ap.add_argument('--bge', default='/data/yanmengyi/sentence_transformer_model/bge-large-en-1.5')
ap.add_argument('--device', default='cuda')
ap.add_argument('--m', type=int, default=3)
ap.add_argument('--out', default='revision_exp/results/expD_routing_consistency.json')
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

Q = pd.read_parquet(os.path.join(a.cache, 'queries.parquet')).reset_index(drop=True)
Q['_row'] = np.arange(len(Q))
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
bge = SentenceTransformer(a.bge, device=a.device)
Emb = bge.encode([strip_template(r['instruction'], r['input']) for _, r in Q.iterrows()],
                 batch_size=64, normalize_embeddings=True, show_progress_bar=False)

# per dataset: eval = first half (probe routing measured here), train = second half (fit router)
def split(ds_list):
    tr, ev = [], []
    for ds in ds_list:
        g = Q[Q['dataset'] == ds].sort_values('qid')
        h = len(g) // 2
        ev += list(g['_row'].values[:h]); tr += list(g['_row'].values[h:])
    return np.array(tr), np.array(ev)

base_top1 = {}; base_topm = {}; rows = []
for n in [3, 6, 9, 12]:
    keep = NESTED[n]
    tr, _ = split(keep)
    clf = LogisticRegression(max_iter=2000, C=10).fit(Emb[tr], Q.loc[tr, 'dataset'].values)
    classes = list(clf.classes_)
    # route the probe eval queries
    agg_top1, agg_jac, agg_orig, cnt = 0.0, 0.0, 0.0, 0
    per_probe = {}
    for ds in PROBE:
        g = Q[Q['dataset'] == ds].sort_values('qid'); h = len(g) // 2
        rows_ev = g['_row'].values[:h]
        proba = clf.predict_proba(Emb[rows_ev])
        top1 = np.array(classes)[proba.argmax(1)]
        order = np.argsort(-proba, 1)[:, :a.m]
        topm = [set(np.array(classes)[o]) for o in order]
        if n == 3:
            base_top1[ds] = top1; base_topm[ds] = topm
        agree = float((top1 == base_top1[ds][:len(top1)]).mean())
        jac = float(np.mean([len(s & b) / len(s | b) for s, b in zip(topm, base_topm[ds][:len(topm)])]))
        orig = float((top1 == ds).mean())   # routed to its own (original) expert
        per_probe[ds] = {'top1_agreement_vs_n3': round(agree, 4), 'top_m_jaccard_vs_n3': round(jac, 4),
                         'routed_to_original': round(orig, 4), 'n_queries': len(top1)}
        agg_top1 += agree * len(top1); agg_jac += jac * len(top1); agg_orig += orig * len(top1); cnt += len(top1)
    rows.append({'n': n, 'top1_agreement_vs_n3': round(agg_top1 / cnt, 4),
                 'top_m_jaccard_vs_n3': round(agg_jac / cnt, 4),
                 'routed_to_original': round(agg_orig / cnt, 4), 'per_probe': per_probe})
    print('n=%2d  top1-agree(vs n3)=%.3f  top%d-jaccard=%.3f  routed-to-original=%.3f'
          % (n, rows[-1]['top1_agreement_vs_n3'], a.m, rows[-1]['top_m_jaccard_vs_n3'], rows[-1]['routed_to_original']))

out = {'probe_tasks': ['ER-amazon-google', 'DI-amazon'], 'baseline_n': 3, 'm': a.m, 'results': rows}
json.dump(out, open(a.out, 'w'), indent=2)
print('SAVED', a.out)
