# P0-3 Bias audit (R1-3): does MELD-DS selection mitigate the augmented pool's distributional bias?
# Pure statistics (no training). ER/amazon-google:
#   pool     = augmented pool (amazon-google-train.json)
#   selected = MELD-DS selected 30% (train-select.json, the PREVIOUS selection result)
#   original = few-shot labeled / init set (train-init.json)
# Reports entity-frequency head/mid/tail buckets, label distribution, and JS divergence vs original.
import json, re, ast, argparse, collections
import numpy as np

def parse_entity(block):
    block = block.strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            d = fn(block)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None

def entities_of(item):
    """Extract the two entity titles from the ER instruction."""
    ins = item['instruction']
    ids = []
    # grab the {...} dict right after 'Entity 1:' and 'Entity 2:'
    for tag in ('Entity 1:', 'Entity 2:', 'Column 1:', 'Column 2:'):
        i = ins.find(tag)
        if i < 0:
            continue
        j = ins.find('{', i)
        if j < 0:
            continue
        depth = 0
        for k in range(j, len(ins)):
            if ins[k] == '{': depth += 1
            elif ins[k] == '}':
                depth -= 1
                if depth == 0:
                    d = parse_entity(ins[j:k + 1])
                    if d:
                        key = d.get('title') or d.get('name') or d.get('column_name') or str(sorted(d.items())[:1])
                        ids.append(str(key).strip().lower())
                    break
    return ids

def label_of(item):
    try:
        d = ast.literal_eval(str(item['output'])); return str(list(d.values())[0]).strip().lower()
    except Exception:
        return str(item['output']).strip().lower()

def bucket_ratio(data, head, mid, tail):
    c = collections.Counter()
    for it in data:
        for e in entities_of(it):
            c['head' if e in head else 'mid' if e in mid else 'tail'] += 1
    tot = sum(c.values()) or 1
    return {b: round(c[b] / tot, 4) for b in ('head', 'mid', 'tail')}

def label_dist(data):
    c = collections.Counter(label_of(it) for it in data); tot = sum(c.values()) or 1
    return {k: round(v / tot, 4) for k, v in sorted(c.items())}

def js_divergence(p, q):
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0) for k in keys]) + 1e-12; pv /= pv.sum()
    qv = np.array([q.get(k, 0) for k in keys]) + 1e-12; qv /= qv.sum()
    m = 0.5 * (pv + qv)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return round(0.5 * kl(pv, m) + 0.5 * kl(qv, m), 4)

ap = argparse.ArgumentParser()
ap.add_argument('--pool', default='train/ER/amazon-google/amazon-google-train.json')
ap.add_argument('--selected', default='train/ER/amazon-google/train-select.json')
ap.add_argument('--original', default='train/ER/amazon-google/train-init.json')
ap.add_argument('--out', default='revision_exp/results/expD_bias_audit.json')
a = ap.parse_args()
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
pool = json.load(open(a.pool)); selected = json.load(open(a.selected)); original = json.load(open(a.original))

# entity frequency over the POOL -> head(top10%) / mid(10-50%) / tail(bottom50%)
freq = collections.Counter()
for it in pool:
    for e in entities_of(it):
        freq[e] += 1
ranked = [e for e, _ in freq.most_common()]
n = len(ranked)
head = set(ranked[:max(1, n // 10)]); mid = set(ranked[n // 10:n // 2]); tail = set(ranked[n // 2:])

# label dist (head-bucket pool/selected for label, plus full)
labp, labs, labo = label_dist(pool), label_dist(selected), label_dist(original)
audit = {
    'task': 'ER-amazon-google',
    'sizes': {'pool': len(pool), 'selected': len(selected), 'original': len(original), 'unique_entities_pool': n},
    'entity_frequency_buckets': {
        'pool': bucket_ratio(pool, head, mid, tail),
        'selected': bucket_ratio(selected, head, mid, tail),
        'original': bucket_ratio(original, head, mid, tail)},
    'label_distribution': {'pool': labp, 'selected': labs, 'original': labo},
    'js_divergence_label': {
        'pool_vs_original': js_divergence(labp, labo),
        'selected_vs_original': js_divergence(labs, labo)},
}
# JS on the entity-bucket distribution too
bp, bs, bo = audit['entity_frequency_buckets']['pool'], audit['entity_frequency_buckets']['selected'], audit['entity_frequency_buckets']['original']
audit['js_divergence_entity_bucket'] = {
    'pool_vs_original': js_divergence(bp, bo), 'selected_vs_original': js_divergence(bs, bo)}

json.dump(audit, open(a.out, 'w'), indent=2)
print(json.dumps(audit, indent=2))
print('\nSAVED', a.out)
