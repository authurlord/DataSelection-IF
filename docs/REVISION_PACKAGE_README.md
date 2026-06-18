# MELD TKDE Revision — package manifest (TKDE-2025-09-3230)

Generated 2026-06-17. All experiments use the paper's own `src/evaluation.py`; the RE eval harness reproduces the paper micro-F1 (0.770), and HF-generate ≡ vLLM greedy verified 100%.

## START HERE
- `docs/REVISION_PER_QUESTION_ANNOTATION.md` — **per-question annotation (问题 | 文章定义/位置 | 回答 | 数据支撑 | 状态)** for every AE/R1/R2 point. The main deliverable.
- `docs/MELDDS_TKDE_response_letter_REVISED.md` — full response-to-reviewers, [TBD]s filled with real data, AE-4 corrected to parity+modularity.
- `docs/REVISION_synthesis.md` — the decisive conclusion: dense ≈ MELD parity is Theorem-2-predicted; reframe to MELD-DS + modularity-under-parity. Read the "FINAL AE-4 VERDICT".

## KEY FINDING (read before using the numbers)
The matched dense multi-task 7B reaches **accuracy parity** with the MELD MoE (not inferiority). This is what Theorem 2 predicts in the full-capacity-7B / small-n regime. The revision succeeds by reframing — MELD-DS (data selection, 32× cheaper than LESS) as the primary contribution + MoE as modularity-under-parity (incremental onboarding, plug-and-play composition) — NOT by claiming MoE accuracy superiority. Do NOT submit "MoE superiority" as the headline.

## figures (revision_exp/results/figs/) — visualizations
NEW (this revision):
- `scaling_dense_vs_meld.png` — task-count scaling, dense reaches parity, no robust interference decay (AE-4/AE-5)
- `selection_flop.png` — selection compute MELD-DS 13.4 vs QuRating 25.3 vs LESS 424.7 PFLOP (AE-3/R2-2.1)
- `compute_tradeoff.png` — continual-onboarding 5.2× cumulative build + 2× inference cost (AE-4)
- `crossds_topm_vs_single.png` — top-m recovers best-single, avoids worst (AE-5/R2-2.4)
EXISTING:
- `expA_combined.png` (ITS / AE-1), `router_scalability.png` (router 3→18 / R1-2,R1-4),
  `moe_crossdataset_topm.png`, `moe_crossds_ER_DC.png`, `moe_merge_vs_serve.png`, `expC_ER.png`

## results (revision_exp/*.txt, *.json)
- `scaling10x_res.txt`, `dense_scaling_ep1.txt` — dense scaling F1 (AE-4)
- `be_res_g*.txt`, `be_redist_g*.txt`, `be_res_local.txt` — batched_eval outputs (cross-dataset + frontier)
- `xds_matrix_results.txt` — cross-dataset single-expert matrix
- `poly_eval_config*.json`, `poly_eval_cfg_*.json` — Poly composable-MoE eval configs
- (Poly per-task F1 q3/mistral — appended when the 12.43 eval finishes)

## code (revision_exp/*.py)
- `batched_eval.py` — load-base-once + swap-LoRA eval engine (the workhorse)
- `poly_train.py`, `poly_eval_infer.py`, `poly_eval_metrics.py`, `bake_poly_to_lora.py`, `bake_verify.py` — Poly composable-MoE
- `expA_*.py`, `expITS.py` — ITS / task-vector subspace (AE-1)
- `expC_inject_noise.py`, `expC_ppl.py`, `expC_eval.py` — noisy-augmentation robustness (AE-6/R1-3)
- `expMoE_*.py` — cross-dataset / router / merge / serve studies (AE-5/R2-2.4)
- `make_revision_figs.py` — regenerates the 4 new figures from on-disk results
- `vllm_eval.py`, `crossds_hf_eval.py`, `strip_cfg.py`, `alignment_check.py` — infra

## docs (full set)
REVISION_PER_QUESTION_ANNOTATION, MELDDS_TKDE_response_letter_REVISED/_draft, REVISION_synthesis,
REVISION_RESULTS, REVISION_REFRAME_core_experiments, RESPONSE_TBD_FILLED, EXPB_RESULTS, EXP_A_log,
MELDDS_TKDE_R1_revision_meeting (AE→§ mapping), MELDDS_TKDE_experiment_plan, MOE_REMEDY_findings,
MOE_BACKUP_SURVEY, RESOURCE_LIMITATIONS.
