# Exp B results (22/22) — Mistral-7B, real src/evaluation.py, HF batched-eval (load-once), bsz4

## Exp 2b — cross-dataset (held-out ER): single-expert vs top-m (KnOTS product-space merge)
| held-out | in-dom | best-1 | worst-1 | mean-1 | top-m | tm−best1 | tm−mean1 |
|---|---|---|---|---|---|---|---|
| abt-buy | 0.893 | 0.908 | 0.467 | 0.797 | 0.900 | −0.008 | +0.103 |
| amazon-google | 0.777 | 0.657 | 0.395 | 0.552 | 0.582 | −0.075 | +0.030 |
| semi-text-c | 0.255 | 0.257 | 0.244 | 0.249 | 0.248 | −0.009 | −0.001 |
| semi-text-w | (na) | 0.769 | 0.635 | 0.703 | 0.751 | −0.019 | +0.047 |
| walmart-amazon | (na) | 0.774 | 0.406 | 0.590 | 0.815 | +0.040 | +0.224 |
| wdc | 0.868 | 0.919 | 0.887 | 0.904 | 0.924 | +0.004 | +0.020 |

**Conclusion:** top-m ≈ best-single (mean tm−best1 ≈ −0.01; amazon-google is the −0.075 outlier), but clearly > mean-single (+0.085 avg) and >> worst-single (abt-buy +0.43, walmart-amazon +0.41). The value of top-m composition is ROBUSTNESS — without knowing which related expert is best, composing recovers ≈best-single and avoids the catastrophic worst case. NOT accuracy superiority over the oracle best single. semi-text-c is a hard dataset (~0.25 for all). Answers AE-5 / R2-2.4 (when standalone+composition helps: held-out / plug-and-play).

## Exp 1 — DC data-efficiency frontier (F1 by selection method; select/w-main = MELD-DS, w-FL/w-IF/w-ppl = signal ablations, w-QuRating = baseline)
| dataset | select | w-main | w-FL | w-IF | w-ppl | w-QuRating |
|---|---|---|---|---|---|---|
| beer | 0.975 | 0.993 | 0.997 | 0.983 | 0.998 | 0.995 |
| hospital | 0.967 | 0.963 | 0.949 | 0.916 | 0.974 | (missing) |
| rayyan | 0.832 | 0.830 | 0.825 | 0.818 | 0.824 | (missing) |

**Conclusion:** MELD-DS (select/w-main) is competitive — BEST on rayyan (select 0.832), top-tier on hospital, near-ceiling on beer. But DC is saturated so margins are small and single-signal ablations (w-ppl, w-FL) sometimes edge out the full method. The frontier story needs (a) the accuracy-vs-%data BUDGET sweep and (b) the SELECTION-COST axis (MELD's 0.5B batch-level selection vs LESS 7B per-sample / QuRating 1.3B full-pool) to be compelling — these are where MELD wins decisively.

## Revision-readiness note
Exp B alone does NOT strongly close AE-4 (dense-parity): cross-dataset = robustness (not accuracy win), DC frontier = small margins. The decisive experiment is Exp 2a (task-count scaling: dense per-task F1 decay vs n + cumulative compute) — IN PROGRESS (dense n=3/5/7 training on 51.11). Full revision review after Exp 2a + the selection-cost FLOP table.

## Infra notes
Run via batched_eval.py (load base once, swap LoRA, unload in try/finally), HF-generate on 51.11 deepspeed env (peft 0.12, after strip_cfg.py removed corda_config), bsz4, [INST] mistral template. 22 jobs across 4 V100 via run_in_background ssh workers (setsid/nohup detach was unreliable). HF≡vLLM greedy verified 100%.
