# Figures for the MoE scalability/router experiments (run LOCALLY after copying results back).
#   Fig A1: router vs oracle vs random task-F1 as expert count n grows  (scalability, R1-4/AE-5)
#   Fig A2: expert utilization + router fit wall-clock as n grows        (stability + complexity, R1-2)
#   Fig B : probe-task F1 vs n for dense / Poly+MHR / MELD               (AE-4 + R2-2.4)
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--sweep', default='revision_exp/results/router_quality/router_quality_sweep.csv')
ap.add_argument('--probe', default='revision_exp/results/expB_probe/probe_f1.csv')
ap.add_argument('--meld_probe', type=float, default=None, help='MELD constant probe-F1 reference line')
ap.add_argument('--out_dir', default='revision_exp/results/moe_figs')
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

# ---------------- Exp A figures ----------------
if os.path.exists(a.sweep):
    s = pd.read_csv(a.sweep).sort_values('n')
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar(s['n'], s['f1_router_mean'], yerr=s['f1_router_std'], marker='o', lw=1.8,
                capsize=3, label='router (content)', color='#1f77b4', zorder=4)
    ax.plot(s['n'], s['f1_oracle_mean'], marker='s', ls='--', lw=1.5, label='oracle (in-domain)', color='#2ca02c')
    ax.plot(s['n'], s['f1_random_mean'], marker='^', ls=':', lw=1.5, label='random', color='#d62728')
    ax.set_xlabel('# experts $n$'); ax.set_ylabel('routed task F1')
    ax.set_title('(A1) Routing quality vs expert count'); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'figA1_scalability.pdf'))
    fig.savefig(os.path.join(a.out_dir, 'figA1_scalability.png'), dpi=200); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    ax1.plot(s['n'], s['util_mean'], marker='o', color='#1f77b4', label='expert utilization (norm. entropy)')
    ax1.fill_between(s['n'], s['util_mean'] - s['util_std'], s['util_mean'] + s['util_std'],
                     alpha=0.18, color='#1f77b4')
    ax1.set_xlabel('# experts $n$'); ax1.set_ylabel('utilization balance', color='#1f77b4')
    ax1.set_ylim(0, 1.05); ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(s['n'], s['fit_s_mean'], marker='s', ls='--', color='#ff7f0e', label='router fit time (s)')
    ax2.set_ylabel('router fit wall-clock (s)', color='#ff7f0e'); ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.set_title('(A2) Stability & training cost vs $n$'); ax1.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'figA2_stability_cost.pdf'))
    fig.savefig(os.path.join(a.out_dir, 'figA2_stability_cost.png'), dpi=200); plt.close(fig)
    print('exp A figs written. sweep rows:', len(s))
else:
    print('no sweep csv yet:', a.sweep)

# ---------------- Exp B figure ----------------
if os.path.exists(a.probe):
    p = pd.read_csv(a.probe)
    # SM-CMS is a degenerate binary-F1 metric (0.7% positive) -> headline = clean ER+DI mean.
    p['erdi'] = (p['ER-amazon-google'] + p['DI-amazon']) / 2.0
    p['method'] = p['tag'].str.extract(r'^(dense|poly|meld)_')
    p['n'] = p['tag'].str.extract(r'_n(\d+)_').astype('Float64')
    sty = {'dense': ('#d62728', 'o', '-', 'dense multi-task'),
           'poly': ('#9467bd', 's', '--', 'Poly+MHR (integrated MoE)'),
           'meld': ('#2ca02c', '^', '-.', 'MELD (modular)')}
    meld_rows = p[p['method'] == 'meld']
    meld_erdi = float(meld_rows['erdi'].mean()) if len(meld_rows) else a.meld_probe
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for meth in ['dense', 'poly']:
        g = p[(p['method'] == meth) & p['n'].notna()]
        if not len(g):
            continue
        gm = g.groupby('n')['erdi'].agg(['mean', 'std']).reset_index()
        c, mk, ls, lab = sty[meth]
        ax.errorbar(gm['n'].astype(float), gm['mean'], yerr=gm['std'].fillna(0), marker=mk, ls=ls,
                    color=c, capsize=3, lw=1.9, label=lab)
    if meld_erdi is not None:
        ax.axhline(meld_erdi, color='#2ca02c', ls='-.', lw=1.9, label='MELD (modular, per-task expert)')
    ax.set_xlabel('# tasks $n$'); ax.set_ylabel('probe-task F1  (ER, DI mean)')
    ax.set_title('(B) Probe-task F1 vs task count\n(SM-CMS excluded: degenerate, 0.7% positive)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'figB_probe_vs_n.pdf'))
    fig.savefig(os.path.join(a.out_dir, 'figB_probe_vs_n.png'), dpi=200); plt.close(fig)
    print('exp B fig written. probe rows:', len(p), 'MELD erdi=%.4f' % (meld_erdi or -1))
else:
    print('no probe csv yet:', a.probe)
print('SAVED ->', a.out_dir)
