# Plot the per-sample perplexity distribution of clean vs flipped (injected-noise) labels.
# NOTE: this is the ORIGINAL (filter-framing) figure -- it plots a single per-sample ppl (proxy + init adapter)
#       and shows clean vs flipped separation. The MELD-DS quality is actually a TRUNCATED RATIO
#       f(PPL_cond/PPL_prior) (paper Eq.9), recomputed in expC_ppl_ratio.py -- prefer that for the revision.
# Inputs:
#   - clean/noised pools: train/ER/amazon-google/amazon-google-train-noise{0,30}.json (from expC_inject_noise.py)
#   - per-sample ppl csv: revision_exp/output/C/ppl_ER_noise30.csv  (from expC_ppl.py: proxy 0.5B + task init adapter)
import json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--clean', default='train/ER/amazon-google/amazon-google-train-noise0.json')
ap.add_argument('--noised', default='train/ER/amazon-google/amazon-google-train-noise30.json')
ap.add_argument('--ppl_csv', default='revision_exp/output/C/ppl_ER_noise30.csv')
ap.add_argument('--out', default='revision_exp/results/figs/expC_ppl_filter.png')
a = ap.parse_args()

clean = json.load(open(a.clean)); noised = json.load(open(a.noised))
flip = np.array([str(clean[i]['output']) != str(noised[i]['output']) for i in range(len(clean))])
ppl = pd.read_csv(a.ppl_csv, index_col=0)['ppl']
idx = ppl.index.values; p = ppl.values; fl = flip[idx]

plt.figure(figsize=(5, 3.4)); bins = np.linspace(0.8, 3, 40)
plt.hist(p[~fl], bins=bins, alpha=0.6, label='clean labels', color='tab:blue', density=True)
plt.hist(p[fl], bins=bins, alpha=0.6, label='flipped (noise)', color='tab:red', density=True)
plt.xlabel('per-sample perplexity (0.5B proxy + init adapter)'); plt.ylabel('density'); plt.legend()
plt.title('C: quality(ppl) separates injected noise (ER, 30%)'); plt.tight_layout()
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
plt.savefig(a.out, dpi=150); print('saved', a.out)
print('clean ppl mean %.3f | flipped ppl mean %.3f' % (p[~fl].mean(), p[fl].mean()))
