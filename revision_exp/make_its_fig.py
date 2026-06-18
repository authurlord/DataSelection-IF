# Publication-quality ITS figure for R1-1 / AE-1 (TKDE revision).
# Two panels: (a) t-SNE of per-task LoRA task vectors with the 3 structurally-different
# non-tabular tasks highlighted (star markers + cosine-to-DP annotation); (b) cumulative
# explained-variance curve showing intrinsic dim (90% var) = 10 over 19 tasks.
# Reuses the exact vector-loading of expITS.py so the geometry matches its.json.
import glob, os, json, argparse
import numpy as np
from safetensors import safe_open
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_vec(path):
    keys = []
    with safe_open(path, 'pt') as f:
        for k in f.keys():
            if 'lora_A' in k or 'lora_B' in k:
                keys.append(k)
        keys.sort()
        parts = [f.get_tensor(k).float().reshape(-1).numpy() for k in keys]
    return np.concatenate(parts)

# pretty labels for the non-tabular points (the R1-1 claim)
NONTAB_PRETTY = {
    'NONTAB-grammar': 'string-transform',
    'NONTAB-mmlu':    'general QA (MMLU)',
    'NONTAB-bbh':     'reasoning (BBH)',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='lora/qwen-0.5B/*/*/init/adapter_model.safetensors')
    ap.add_argument('--out_dir', default='revision_exp/results/ITS_v3')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(a.glob))
    names = [f.split('lora/qwen-0.5B/')[1].split('/init')[0].replace('/', '-') for f in files]
    # drop the AG-init bake helpers if present (not real task vectors of interest)
    vecs = []; keep = []; D = None
    for f, n in zip(files, names):
        if n.startswith('AG-init'):
            continue
        try:
            v = load_vec(f)
            if D is None: D = len(v)
            if len(v) != D:
                print('skip dim-mismatch', n, len(v)); continue
            vecs.append(v); keep.append(n)
        except Exception as e:
            print('skip', n, e)
    X = np.stack(vecs); names = keep
    print('task vectors:', X.shape, 'tasks:', len(names))

    is_nt = np.array([n.startswith('NONTAB') for n in names])
    # cosine of each non-tabular point to the DP (tabular) cluster
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    dp_idx = np.where(~is_nt)[0]
    nt_cos = {names[i]: float(S[i, dp_idx].mean()) for i in np.where(is_nt)[0]}
    print('non-tabular cosine-to-DP:', {k: round(v, 3) for k, v in nt_cos.items()})

    # PCA intrinsic dim (full curve)
    p = PCA(n_components=min(len(names), X.shape[0])).fit(X)
    ev = np.cumsum(p.explained_variance_ratio_)
    dim90 = int(np.searchsorted(ev, 0.90) + 1)
    dim95 = int(np.searchsorted(ev, 0.95) + 1)
    print('INTRINSIC DIM 90%%=%d 95%%=%d' % (dim90, dim95))

    # t-SNE (deterministic) -- same params as expITS for reproducibility
    Z = TSNE(n_components=2, perplexity=min(5, len(names) - 1),
             random_state=0, init='pca').fit_transform(X)

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={'width_ratios': [1.55, 1]})

    # panel (a): t-SNE
    tasks = [n.split('-')[0] for n in names]
    dp_fams = sorted(set(t for t, nt in zip(tasks, is_nt) if not nt))
    cmap = plt.cm.tab10
    fam_color = {t: cmap(i % 10) for i, t in enumerate(dp_fams)}
    NT_COLOR = '#d62728'

    for j, n in enumerate(names):
        if is_nt[j]:
            axL.scatter(Z[j, 0], Z[j, 1], marker='*', s=340, color=NT_COLOR,
                        edgecolors='black', linewidths=1.0, zorder=5)
        else:
            axL.scatter(Z[j, 0], Z[j, 1], marker='o', s=85,
                        color=fam_color[tasks[j]], edgecolors='white',
                        linewidths=0.6, zorder=3)

    xmin, xmax = Z[:, 0].min(), Z[:, 0].max()
    rng = xmax - xmin
    axL.set_xlim(xmin - 0.22 * rng, xmax + 0.55 * rng)   # extra right room for non-tab labels
    ymin, ymax = Z[:, 1].min(), Z[:, 1].max()
    axL.set_ylim(ymin - 0.12 * (ymax - ymin), ymax + 0.12 * (ymax - ymin))
    cx = (xmin + xmax) / 2

    # manual placement overrides (dx, dy, ha) for points that would collide;
    # keyed by name. Far-right MMLU/BBH stars get pushed right with v-separation.
    NT_POS = {
        'NONTAB-mmlu':    (3.0,  6.0, 'left'),
        'NONTAB-bbh':     (3.0, -6.0, 'left'),
        'NONTAB-grammar': (3.0,  0.0, 'left'),
    }
    for j, n in enumerate(names):
        right = Z[j, 0] <= cx          # label on the side with more room
        ha = 'left' if right else 'right'
        dx = 2.2 if right else -2.2
        if is_nt[j]:
            ddx, ddy, ha = NT_POS.get(n, (dx, 0.0, ha))
            lab = '%s (cos %.2f)' % (NONTAB_PRETTY.get(n, n), nt_cos[n])
            axL.annotate(lab, (Z[j, 0], Z[j, 1]),
                         xytext=(Z[j, 0] + ddx, Z[j, 1] + ddy),
                         fontsize=8.5, fontweight='bold', color=NT_COLOR,
                         ha=ha, va='center', zorder=6,
                         arrowprops=dict(arrowstyle='-', color=NT_COLOR, lw=0.7))
        else:
            axL.annotate(n, (Z[j, 0], Z[j, 1]), xytext=(Z[j, 0] + dx, Z[j, 1]),
                         fontsize=7.5, ha=ha, va='center', color='#333333')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=fam_color[t],
                      markersize=9, label=t) for t in dp_fams]
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=NT_COLOR,
                          markeredgecolor='black', markersize=15,
                          label='non-tabular'))
    axL.legend(handles=handles, fontsize=8.5, ncol=2, loc='upper left',
               framealpha=0.95, columnspacing=0.8, handletextpad=0.3)
    axL.set_title('(a) Per-task LoRA task vectors (t-SNE)', fontsize=11)
    axL.set_xlabel('t-SNE dim 1', fontsize=9); axL.set_ylabel('t-SNE dim 2', fontsize=9)
    axL.tick_params(labelsize=8)

    # panel (b): cumulative explained variance
    axR.plot(range(1, len(ev) + 1), ev, 'o-', color='#1f77b4', markersize=5, lw=1.6)
    axR.axhline(0.90, ls='--', c='gray', lw=1.0)
    axR.axvline(dim90, ls=':', c=NT_COLOR, lw=1.2)
    axR.annotate('90%% var\n@ %d dims' % dim90, (dim90, 0.90),
                 xytext=(dim90 + 1.5, 0.62), fontsize=9, color=NT_COLOR,
                 arrowprops=dict(arrowstyle='->', color=NT_COLOR, lw=1.0))
    axR.set_xlabel('# principal components', fontsize=9)
    axR.set_ylabel('cumulative explained variance', fontsize=9)
    axR.set_title('(b) Intrinsic dimensionality (19 tasks)', fontsize=11)
    axR.tick_params(labelsize=8); axR.set_ylim(0.2, 1.02)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = os.path.join(a.out_dir, 'its_figure.' + ext)
        fig.savefig(out, dpi=300, bbox_inches='tight'); print('SAVED', out)
    plt.close(fig)

if __name__ == '__main__':
    main()
