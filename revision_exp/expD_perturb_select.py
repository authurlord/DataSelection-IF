# P0-1 (option 2, gradient-free): produce MELD-DS-style DIVERSITY selections under perturbed clustering.
# Variable = batch/cluster cohesion (perturb fraction p). For each p: cluster RAG embeddings, perturb the
# cluster assignment, then select the 30% budget that best covers the (perturbed) clusters (representatives
# closest to their cluster centroid). p=0 -> tight semantic clusters -> diverse coverage; p=1 -> random
# clusters -> centroids drift to dense regions -> selection loses diversity. No gradients (batch-IF dropped;
# this isolates the F_div / coverage path, consistent with the 'diversity term compensates' narrative).
import os, json, argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def perturb_clusters(labels, p, rng):
    labels = labels.copy(); n = len(labels)
    k = int(round(p * n))
    if k == 0:
        return labels
    idx = rng.choice(n, k, replace=False)
    sh = labels[idx].copy(); rng.shuffle(sh)
    labels[idx] = sh
    return labels

def select_under_clustering(E, labels, budget, K):
    """select `budget` samples closest to their cluster centroid (representative coverage)."""
    cent = np.zeros((K, E.shape[1]), np.float32)
    for c in range(K):
        m = labels == c
        cent[c] = E[m].mean(0) if m.any() else 0.0
    d = np.linalg.norm(E - cent[labels], axis=1)        # dist to own (perturbed) cluster centroid
    return np.argsort(d)[:budget]                        # most representative overall

def coverage(E, idx, K, rng):
    """diversity proxy of the selected set: # distinct kmeans-cells covered (on a fixed reference clustering)."""
    km = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=1024).fit(E)
    return len(set(km.labels_[idx])) / K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='train/ER/amazon-google/amazon-google-train.json')
    ap.add_argument('--bge', default='/data/yanmengyi/sentence_transformer_model/bge-large-en-1.5')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--budget', type=float, default=0.30)
    ap.add_argument('--K', type=int, default=300)
    ap.add_argument('--ps', default='0,0.25,0.5,0.75,1.0')
    ap.add_argument('--out_dir', default='revision_exp/seed_study/expD_select')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    pool = json.load(open(a.pool))
    n = len(pool); B = int(round(a.budget * n))
    txt = [(p['instruction'] + ' ' + str(p.get('input', ''))) for p in pool]
    # entity content only (drop fixed template header) for a meaningful embedding
    def content(s):
        if 'Output format example' in s:
            s = s.split('Output format example', 1)[1]
            s = s.split('\n\n', 1)[1] if '\n\n' in s else s
        return s[:1000]
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer(a.bge, device=a.device)
    E = bge.encode([content(t) for t in txt], batch_size=64, normalize_embeddings=True,
                   show_progress_bar=False).astype(np.float32)
    print('pool %d, budget %d, embed %s' % (n, B, E.shape))
    base = MiniBatchKMeans(a.K, random_state=0, n_init=3, batch_size=1024).fit(E)
    labels0 = base.labels_
    summary = []
    for p in [float(x) for x in a.ps.split(',')]:
        rng = np.random.default_rng(0)
        labels = perturb_clusters(labels0, p, rng)
        idx = select_under_clustering(E, labels, B, a.K)
        cov = coverage(E, idx, a.K, rng)
        sel = [pool[i] for i in sorted(idx.tolist())]
        fn = os.path.join(a.out_dir, 'sel_p%d.json' % int(p * 100))
        json.dump(sel, open(fn, 'w'), ensure_ascii=False)
        summary.append({'p': p, 'n_selected': len(sel), 'coverage': round(float(cov), 4)})
        print('p=%.2f  selected=%d  coverage=%.4f  -> %s' % (p, len(sel), cov, fn))
    json.dump(summary, open(os.path.join(a.out_dir, 'select_summary.json'), 'w'), indent=1)
    print('SELECT DONE')

if __name__ == '__main__':
    main()
