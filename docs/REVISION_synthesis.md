# MELD TKDE Revision — Synthesis (4-agent investigation: FLOPS, paper/Theorem-2 bottleneck, literature protocols)

## THE decisive conclusion (all four investigations converge)
In-distribution accuracy parity (dense-7B ≈ MELD MoE, dense +0.01) is **NOT a refutation** — it is exactly what **Theorem 2 predicts** in the high-N / small-n regime, and the modular design is defended on **compute/scaling axes the dense model structurally cannot win**. The single experiment that closes the revision:

> **Compute-matched task-count scaling (Exp 2a): per-task F1 AND cumulative training compute vs n ∈ {3,5,7,10}.**
> Dense = retrain on the union at each n (compute superlinear; per-task F1 decays from cross-task interference). MELD = +1 isolated expert per task (constant marginal cost; per-task F1 flat). It turns AE-4 parity from a contradiction into a *Theorem-2 confirmation* and simultaneously answers AE-5.

## Theorem 2 — what it actually licenses (paper re-read)
Theorem 2 separates dense vs MoE error bounds ONLY when: (a) model capacity small (KL term), (b) per-domain N small, (c) task count n grows under fixed sparsity k. With a full-capacity 7B + adequate data at small n, the dense bound is tight → **parity is predicted**. The abstract's "theoretical proofs for the superiority of the MoE design" over-generalized this. **Action: downgrade the abstract/VI-B accuracy-superiority claim to a generalization-bound-in-n/N-regime + marginal-cost/extensibility claim; re-anchor headline on MELD-DS (data selection).**

## Rigorous FLOPS (Mistral-7B, LoRA r16 all-linear, 5 epochs; union=31,166 ex = exact concat of 16 subsets)
LoRA training cost uses active params = base 7.24B + LoRA 41.94M (frozen base still does full fwd+bwd). Avg seq 312 tok.
| Axis | Dense (1 LoRA on union) | MELD MoE (16 experts) | Verdict |
|---|---|---|---|
| **Total training FLOPS (5 ep)** | 2.13e18 | 2.13e18 (Σ subsets) | **IDENTICAL** (≤0.6%) — no first-build advantage |
| **Inference / query (top-2)** | 1.0× (1×7B) | **2.0×** (2 full-7B experts) | MELD **more expensive** at serving (m× for top-m); router negligible. State honestly. |
| **Marginal cost to add 1 task** | full union retrain = 16× | **1×** (1 expert on its ~1/16 subset) | **MELD ≈ N× (≈16×) cheaper**; up to 60× for small tasks; dense super-linear over k additions |
**Defensible takeaway:** MELD trades a constant ~2× inference overhead for order-N cheaper, isolated, forgetting-free task extension — an amortization win when the task set evolves, NOT a per-query efficiency win.

## Four defense axes (literature-backed) + which to lead with
1. **Task interference grows with n (STRONGEST)** — dense per-task F1 decays as n↑ (gradient conflict / negative transfer); modular flat. Metrics: mean per-task F1 vs n; BWT/Forgetting/ACC matrix (Lopez-Paz GEM 2017). → Exp 2a. **Lead here.**
2. **Marginal cost / plug-and-play extensibility** — O(1) add-task, no data replay, decentralized, privacy, modular debugging (Arrow/Ostapenko COLM'24; MoErging survey 2408.07057; PHATGOOSE ICML'24). Metric: cumulative GPU-h & data-touched vs #tasks. → uses FLOPS table above.
3. **Data-budget frontier (MELD-DS contribution)** — accuracy vs %data, methods = MELD-DS vs random/full/DSIR/SuperFilter/QuRating + own-signal ablations (diversity/quality/influence). Anchor: full-data line; headline: small % matches full (LESS ICML'24 template; DSIR NeurIPS'23). → Exp 1 frontier.
4. **Iso-FLOPS / iso-active-param Pareto** — weaker for MELD (top-m = m full 7B, not sub-FFN sparsity), use cautiously.

## Cross-dataset (corroboration, NOT headline; honest)
Held-out ER: top-m KnOTS-merge avg 0.800 vs arbitrary-single 0.630; 6/7 ≥ single. BUT top-m ≈ *average* single, does NOT beat *best* single; held-out dense still generalizes better (0.78 vs 0.58). Frame as **plug-and-play robustness to not knowing the right expert**, not accuracy superiority.

## AE-1..6 status (from paper+docs)
- **AE-1** (ITS beyond tabular): CLOSED — non-tabular task co-embeds (dim stays 8, cos 0.945).
- **AE-2** (Thm 6 σ): CLOSED + forces correction — error rises monotonically with σ; σ(|B|) increases with |B| (not constant) → restate Thm 6 as σ(|B|)/√|B| tradeoff (explains Table V U-shape).
- **AE-3** (clustering cost): CLOSED — clustering 1.8–4.3% of selection time.
- **AE-4** (matched dense-7B): OPEN/adverse on accuracy → **close via Exp 2a reframe** (parity = Theorem-2-predicted; win on scaling/marginal-cost).
- **AE-5** (routing stability/scalability/standalone): PARTIAL — router stable 3→18 experts; **scaling-cost curve (Exp 2a) is the missing piece**.
- **AE-6** (noise robustness): CLOSED — ppl-filtering captures 94.3% injected noise.

## Setup audit (agent 1) — fixes to apply
- Both Poly runs healthy (~13-15h ETA, 5ep, per-epoch ckpt). Matrix/master/4×V100 OK.
- Bugs: (a) hospital/rayyan `w-QuRating` adapters MISSING → frontier holes (2 cells); (b) workers use `2>/dev/null` → job crashes become silent zero-results (assert expected line counts after each phase).

## Immediate priority shift
Exp 2a (task-count scaling) is promoted to **#1** (was #4) — it is the closing experiment. Train dense at n={3,5,7,10} via llama-factory (5ep) + eval per-task F1; MELD modular = eval existing experts. Pair with the FLOPS marginal-cost table.

## SELECTION-COST accounting (CRITICAL for the frontier — do NOT omit)
The data-efficiency frontier (Exp 1) must compare accuracy per (SELECTION + TRAINING) compute, not training alone. Selection-stage cost per method:
- **LESS**: warmup LoRA train + PER-SAMPLE LoRA gradient (fwd+bwd) over the FULL pool on the target 7B + gradient datastore. ~ |pool| * 6 * N_7B * tokens (per-sample backward) — the most expensive.
- **QuRating**: a trained ~1.3B rater run over the FULL pool (inference). ~ |pool| * 2 * N_1.3B * tokens.
- **SuperFiltering**: IFD/perplexity (a few model forwards) over the full pool.
- **DSIR**: n-gram hashing, no model → ~0 FLOPS.
- **MELD-DS**: FL clustering (embed pool once) + PPL on **0.5B proxy** + **batch-level** IF on 0.5B (|pool|/|B| batches, fwd+bwd) → paper's "3–10% of traditional IF". KEY: 0.5B proxy + batch-level (not 7B per-sample) makes MELD selection ~order(s) cheaper than LESS/QuRating.
=> Reframe strength: MELD-DS reaches the same accuracy-vs-data-budget frontier at a FRACTION of the SELECTION compute (0.5B/batch vs 7B/per-sample for LESS, 1.3B/full-pool for QuRating). This operationalizes the batch-IF claim and answers AE-3 (full cost breakdown incl. selection). Build a per-method selection-FLOPS table when synthesizing Exp 1.

## FINAL AE-4 VERDICT (Exp 2a scaling result + codex review — DECISIVE)
Task-count scaling did NOT robustly show dense interference: mean per-task F1 — dense 0.5ep(10x): 0.860→0.880 (no decay, ≈MELD 0.892); dense 1ep: 0.840→0.811 (decays, but a TRAINING-STAGE-DEPENDENT TRANSIENT); dense 5ep (Exp B): ≈MELD parity. => The "MoE beats dense via interference" claim is a compute-confounded artifact; weakening dense compute does NOT create an honest accuracy win (it ERASES the signal). DO NOT pursue further dense-crippling — it is non-robust AND dishonest.
**Honest, codex-validated revision strategy (de-emphasize MoE superiority; elevate MELD-DS + modularity-under-parity):**
1. MELD-DS (PRIMARY): efficient data selection, selection-compute 32x cheaper than LESS, 1.9x vs QuRating.
2. Matched dense-7B = PARITY in static in-distribution (scientific clarification, NOT failed hypothesis): "MELD's MoE is not an accuracy amplifier in the small-task full-capacity-7B regime; it is a modular alternative that maintains accuracy while reducing incremental cost + enabling plug-and-play transfer."
3. MoE serving = lifecycle modularity UNDER PARITY: incremental task onboarding (no full-union retrain / no cross-task regression), per-expert debug/version, cross-dataset top-m composition (≈best-single, >mean-single, robust). State inference cost ~2x PLAINLY.
4. 5.2x cumulative-build compute = ONLY as continual-onboarding scenario (concede: all-tasks-known-upfront → dense = same cost). Theorem-2 alignment: separation predicted only in low-capacity/low-N/large-n regime → parity here is expected.
**Bottom line:** revision succeeds via reframing (parity is honest + expected), NOT via proving MoE accuracy superiority. If the paper keeps "MoE superiority" as headline → easy reject.
