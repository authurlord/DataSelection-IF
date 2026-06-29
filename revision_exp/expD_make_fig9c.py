# P0-1 aggregate: join downstream F1 (per perturbation p) with the existing sensitivity curve
# (sigma_norm, nMAE) -> results/expD_cohesion_f1.json + Fig 9(c) (downstream F1 vs perturbation p).
import json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--f1_csv', default='revision_exp/results/expD_cohesion_f1_raw.csv')
ap.add_argument('--sens', default='/tmp/ER_sensitivity.json')
ap.add_argument('--out_json', default='revision_exp/results/expD_cohesion_f1.json')
ap.add_argument('--out_fig', default='revision_exp/results/figs/fig9c_cohesion_f1.png')
a = ap.parse_args()
import os; os.makedirs(os.path.dirname(a.out_fig), exist_ok=True)

f1 = pd.read_csv(a.f1_csv)
f1['p'] = f1['tag'].str.extract(r'p(\d+)').astype(int) / 100.0
f1 = f1.sort_values('p')
sens = json.load(open(a.sens))['error_vs_sigma_B8']
sp = np.array([e['p'] for e in sens]); ssig = np.array([e['sigma_norm'] for e in sens])
snm = np.array([e['error']['proposed'] for e in sens])

rows = []
for _, r in f1.iterrows():
    p = float(r['p'])
    rows.append({'perturbation': p,
                 'sigma_norm': round(float(np.interp(p, sp, ssig)), 4),
                 'nMAE': round(float(np.interp(p, sp, snm)), 4),
                 'downstream_f1': round(float(r['f1']), 4)})
out = {'task': 'ER-amazon-google', 'note': 'option-2 gradient-free diversity selection under perturbed '
       'clustering; sigma_norm/nMAE interpolated from the Fig.9(a) sensitivity curve.', 'conditions': rows}
json.dump(out, open(a.out_json, 'w'), indent=2)
print(json.dumps(out, indent=2))

ps = [r['perturbation'] for r in rows]; f1s = [r['downstream_f1'] for r in rows]; nm = [r['nMAE'] for r in rows]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(ps, f1s, 'o-', color='#2563eb', lw=2, label='downstream F1 (ER)')
ax.set_xlabel('cluster perturbation ratio $p$'); ax.set_ylabel('downstream F1', color='#2563eb')
ax.tick_params(axis='y', labelcolor='#2563eb'); ax.set_ylim(min(f1s) - 0.03, max(f1s) + 0.03)
ax2 = ax.twinx()
ax2.plot(ps, nm, 's--', color='#dc2626', lw=1.6, label='batch-IF approx error (nMAE)')
ax2.set_ylabel('batch-IF approx error (nMAE)', color='#dc2626'); ax2.tick_params(axis='y', labelcolor='#dc2626')
ax.set_title('(c) Downstream F1 vs batch cohesion (ER)'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(a.out_fig, dpi=200); fig.savefig(a.out_fig.replace('.png', '.pdf'))
print('SAVED', a.out_json, a.out_fig)
