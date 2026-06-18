# MELD MoE Scalability / Router / Multi-task Comparison (+ ITS) — Result Package

TKDE revision experiments answering **R1-2** (standalone-router training complexity + routing stability),
**R1-4 / AE-5** (scalability as #tasks grows), **AE-4** (matched dense multi-task 7B baseline),
**R2-2.4** (standalone-modular vs integrated MoE), and **R1-1 / AE-1** (intrinsic task subspace beyond
tabular — see `its/`). Base model: Mistral-7B-Instruct-v0.2; ITS task-vectors use Qwen2.5-0.5B LoRAs.

Generated 2026-06-18. Self-contained: figures, plotting/pipeline scripts, raw data, configs.

---

## 1. Results at a glance

### Exp A — router n-sweep (one sweep → scalability + stability + complexity)
`figs/figA1_scalability.png`, `figs/figA2_stability_cost.png` ; data `data/router_quality_sweep.csv`.
n = 3..12 experts, 3 seeds. The content router (BGE embedding of the instance content → logistic
regression over the kept experts; trained on a held-out split, no train==test leakage) is compared to an
**oracle** (always the in-domain expert) and a **random** router, measured as real downstream task-F1.

- **Scalability (R1-4/AE-5):** router-F1 ≈ oracle-F1 at every n (gap ≤ 0.017) and ≫ random (+0.15–0.20).
  Routing quality does NOT degrade as the expert pool grows to 12.
- **Stability (R1-2):** expert utilization (normalized entropy) ~0.98 (no collapse onto a few experts);
  cross-seed F1 std shrinks as n grows (0.23 → 0.00).
- **Complexity (R1-2):** router fit is seconds (a logistic-regression solver jump to ~13–15 s at n ≥ 10,
  still seconds), trainable params linear in n, and the router NEVER retrains the experts.

### Exp B — probe-task F1 vs task count (dense vs Poly+MHR vs MELD)
`figs/figB_probe_vs_n.png` ; data `data/probe_f1_merged.csv`.
Fixed probe = {ER amazon-google, DI amazon, SM CMS}; nested unions n = 3/6/9/12; all methods trained 1 epoch.
Headline metric = **ER+DI mean** (SM-CMS excluded, see caveat). dense = 2 seeds (s0,s1); poly = 1 seed;
MELD = per-task experts (constant reference).

| n | dense (ER+DI mean ±std) | Poly+MHR | MELD (const) |
|---|---|---|---|
| 3 | 0.736 ± 0.000 | 0.702 | 0.722 |
| 6 | 0.710 ± 0.000 | 0.686 | 0.722 |
| 9 | 0.711 ± 0.003 | 0.667 | 0.722 |
| 12 | 0.715 (1 seed) | 0.696 | 0.722 |

- **MELD (modular) is flat ≈ 0.722** across n — per-task experts do not interfere.
- **dense ≈ MELD (parity)**: starts ≥ MELD at n=3, settles just below (within ~0.01); tight 2-seed error
  bars confirm the small decline is real, not noise → supports **AE-4** (no MoE accuracy superiority; parity).
- **Poly+MHR (integrated MoE) is lowest at every n** (~0.67–0.70) → supports **R2-2.4** (the standalone
  modular design is at least as stable as, and here above, an integrated MoE as task count grows).

### Honest caveats
- **SM-CMS is a degenerate metric**: 36 positives / 5127 (0.7%); binary F1(pos=match) drives ALL three
  methods (incl. the dedicated MELD SM expert) to ≈ 0. It is excluded from the headline ER+DI mean and
  reported only in the raw CSVs. DI is flat ~0.65–0.69 for all three (parity); ER carries the trend.
- **Poly is single-seed**: poly seed-1 (error bars) was attempted but dropped due to HuggingFace-Hub
  flakiness on the training host (`LocalEntryNotFoundError` mid-train). dense's 2-seed error bars suffice
  to establish the trend is real.

---

## 2. Files

```
figs/      figA1_scalability.{png,pdf}      Exp A: routed F1 (router/oracle/random) vs n
           figA2_stability_cost.{png,pdf}   Exp A: utilization + router fit wall-clock vs n
           figB_probe_vs_n.{png,pdf}        Exp B: probe ER+DI mean F1 vs n (dense/Poly+MHR/MELD)
data/      router_quality_sweep.csv/.json   Exp A per-n: f1_{router,oracle,random}, util, route_acc, fit_s, n_params (mean±std over seeds)
           probe_f1_merged.csv              Exp B per (method,n,seed): ER / DI / SM-CMS F1 + probe_mean
           probe_f1_dense_meld_raw.csv      raw dense+MELD eval rows (51.11)
           probe_f1_poly_raw.csv            raw poly eval rows (12.43)
scripts/   expMoE_prep.py                   build manifest + dense unions + poly per-task files
           expMoE_router_quality.py         Exp A: --phase predict (vLLM) then --phase route (n-sweep)
           expMoE_eval_probe.py             Exp B probe-F1: --mode {dense|poly|meld}, full-test, rescue()
           poly_train.py                    PEFT PolyConfig + MHR (n_splits) training (--seed for multi-seed)
           make_moe_figs.py                 produce figA1/figA2/figB from the CSVs
           {dense,poly,poly_eval_full,poly_s1}_driver.sh   train+eval drivers (one nohup per ssh)
configs/   manifest.csv                     12 discriminative experts (task,dataset,adapter,test_file)
           dataset_info_probe.json          LLaMA-Factory dataset registration for the nested unions
           dense-probe.yaml, _dense_n*_s*.yaml   dense LoRA training configs (1 epoch, fp16, mistral)
           synthea-expert.yaml              per-task MELD expert config for SM/synthea
its/       its_figure.{png,pdf}             ITS (R1-1/AE-1): t-SNE of 19 per-task LoRA task vectors +
                                            intrinsic-dim curve; the 3 structurally-different non-tabular
                                            points (string-transform / general-QA MMLU / reasoning BBH)
                                            co-embed with the tabular DP cluster (cos 0.945/0.967/0.966)
           its_varexp.png                   cumulative explained variance (intrinsic dim 90% = 10 / 19 tasks)
           its.json                         cosine matrix + intrinsic-dim numbers
           make_its_fig.py                  regenerate its_figure from the qwen-0.5B LoRA adapters
           ITS_v3_results.md                ITS write-up + provenance
```

### ITS (R1-1 / AE-1) — intrinsic task subspace holds beyond tabular
`its/its_figure.png`: 19 task vectors = 16 tabular DP + 3 structurally-different non-tabular tasks
(synthetic string-transformation, general-domain QA = MMLU, multi-step reasoning = BBH). Intrinsic
dimensionality (90% variance) = 10; the 3 non-tabular task vectors co-embed with the tabular cluster
(cosine 0.945 / 0.967 / 0.966) rather than appearing as outliers — geometric evidence that the ITS
assumption extends to structurally different domains. (Primary R1-1 evidence is the OOD MMLU/BBH
performance study elsewhere in the revision; this is the geometric auxiliary.)

---

## 3. Reproduction

Machines: **51.11** (4×V100, env `deepspeed`, vLLM 0.5.4) for dense + MELD + Exp A; **12.43**
(A800 cuda6/7 only, env `deepspeed`, no flash-attn → sdpa) for Poly+MHR. Adapters/base are large and
machine-local; paths in `configs/` and the driver scripts.

```bash
# 0) build data artifacts (manifest, dense unions, poly per-task files)
python scripts/expMoE_prep.py
#    register the union_n*probe datasets into LLaMA-Factory data/dataset_info.json (see dataset_info_probe.json),
#    and set dataset_dir to that data/ dir in the dense yamls.

# 1) Exp A (51.11): build prediction matrix once on GPU, then sweep on CPU
python scripts/expMoE_router_quality.py --phase predict --manifest configs/manifest.csv \
    --base <Mistral-7B> --per_ds 300 --cache results/router_quality/predmatrix
python scripts/expMoE_router_quality.py --phase route --manifest configs/manifest.csv \
    --bge <bge-large-en-1.5> --cache results/router_quality/predmatrix --ns 3,4,5,6,7,8,9,10,11,12 --seeds 0,1,2 \
    --out results/router_quality
#    (the SM/synthea expert, 12th pick, is trained via configs/synthea-expert.yaml on 51.11 first)

# 2) Exp B
#    dense (51.11): bash scripts/dense_driver.sh <gpu> <n...>      (trains _dense_nN_sS.yaml, 1 epoch)
#    Poly+MHR (12.43): bash scripts/poly_driver.sh <cuda6|7> <n...>  (poly_train --n_skills 8 --n_splits 4 --epochs 1 --bsz 2 --cutoff 1024)
#    full-test eval (cap 6000):
python scripts/expMoE_eval_probe.py --mode dense --base <Mistral> --adapter <dense_adapter> --tag dense_nN_sS --cap 6000 --out results/expB_probe/probe_f1_full.csv
python scripts/expMoE_eval_probe.py --mode poly  --base <Mistral> --poly <poly_adapter>   --tag poly_nN_s0  --cap 6000 --out results/expB_probe/probe_f1_full.csv   # direct PEFT (MHR cannot be baked)
python scripts/expMoE_eval_probe.py --mode meld  --base <Mistral> --tag meld_s0 --cap 6000 --out results/expB_probe/probe_f1_full.csv   # per-task experts

# 3) figures
python scripts/make_moe_figs.py --sweep data/router_quality_sweep.csv --probe data/probe_f1_merged.csv --out_dir figs
```

### Gotchas baked into the scripts
- `expMoE_eval_probe.py` / `expMoE_router_quality.py` include `rescue()` + `er_sm_safe()`: Poly emits
  truncated dicts (`'mismatch'}`) and `src/evaluation.py::Transfer()` scores any ER/SM pred lacking
  "mismatch" as a positive match — both are normalized so a garbage prediction is never counted correct.
- The route phase splits each dataset (eval = first half, train = second half); the first-half keeps RE's
  positional `test_RAG.csv` gold aligned, and train≠eval removes router leakage.
- Eval uses the full probe test (`--cap 6000`): a first-N subsample makes the 0.7%-positive SM-CMS metric
  degenerate.
