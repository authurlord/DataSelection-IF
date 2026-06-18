# MELD TKDE Revision — HANDOFF & Reproduction (as of 2026-06-18)

Single source of truth to resume the revision work. Branch: `revision-results-2026-06-18` (pushed). Session transcript exported to `revision_exp/session_exports/`.

---

## 0. STATE — what is DONE
| Deliverable | Status | Doc / artifact |
|---|---|---|
| Poly composable-MoE per-task eval (q3 + mistral, 15 datasets) | ✅ | `results/poly_{qwen3_4b,mistral_7b}_eval_norm.md` |
| Paper-MELD vs Poly comparison | ✅ | `docs/POLY_vs_PAPER_comparison.md` |
| dense OOD-on-all-DP eval (dense-n7-10x, 15 datasets) | ✅ | `results/dense_n7_10x_eval.md` |
| **dense epoch progression ep1-5 (15 DP + OOD)** | ✅ | `docs/DENSE_epoch_progression.md` |
| **OOD matrix base / dense / Poly (MMLU+BBH)** | ✅ | `docs/REVISION_OOD_R1-1_results.md` |
| **ITS v3 (16 DP + grammar/mmlu/bbh, reproducible)** | ✅ | `docs/ITS_v3_results.md`, `results/ITS_v3/` |
| Revision per-question annotation + requirements review | ✅ | `docs/REVISION_PER_QUESTION_ANNOTATION.md`, `docs/REVISION_REQUIREMENTS_REVIEW.md` |
| Response letter (filled, AE-4 corrected) | ✅ | `docs/MELDDS_TKDE_response_letter_REVISED.md` |
| FLOP / synthesis / Exp B | ✅ | `docs/REVISION_synthesis.md`, `docs/EXPB_RESULTS.md` |

## 0b. KEY HONEST FINDINGS (do not regress these)
1. **AE-4: matched dense ≈ MELD parity in-distribution (NOT MELD-superior).** Reframe to modularity-under-parity. Dense-crippling was tried and rejected (non-robust + dishonest).
2. **R1-1 OOD (clean 5ep schedule)**: dense MMLU 57.99→55.30 / BBH 46.26→41.44 over ep1→ep5 (forgetting, then plateau; BBH stays > base). DP rises (ER 66→90s). MELD/Poly modular (frozen base + detachable LoRA) preserves OOD: Poly-ER MMLU 58.16 / BBH 43.79 ≥ base (57.71 / 36.43). No catastrophic collapse because LoRA freezes base (the I-LoRA full-FT collapse does not apply).
3. **ITS holds beyond tabular**: 19-task intrinsic dim 90%=10; grammar/mmlu/bbh co-embed (cos 0.945/0.967/0.966). Geometric aux evidence; OOD perf is the primary R1-1 evidence.
4. **MELD-DS selection compute 32× cheaper than LESS** (13.4 vs 424.7 PFLOP), 1.9× vs QuRating.

---

## 1. MACHINES / ENVS
- **12.43** (wangys): A800-80G ×8, **use cuda6/7 ONLY**. vLLM eval env `vllm_pascal` (vLLM 0.19.1). llama-factory in `/data/home/wangyaoshu/anaconda3/envs/deepspeed`. meld_revision dir `/data/home/wangys/meld_revision`. Base models in `/data/home/wangys/model/`.
- **51.11** (yanmengyi): V100-32G ×4, all usable. llama-factory in conda `deepspeed`; repo `/data/yanmengyi/DataSelection-IF`; LLaMA-Factory at `/data/yanmengyi/LLaMA-Factory`. Mistral base `/data/yanmengyi/huggingface/Mistral-7B-Instruct-v0.2`.
- **51.10** (LOCAL, this repo `/home/yanmy/DataSelection-IF`): RTX 3090, **cuda2 only**, release immediately. base env: torch2.11/tf4.57/peft0.18. Qwen2.5-0.5B copied to `/home/yanmy/models/Qwen2.5-0.5B-Instruct`.

**CRITICAL GPU discipline** (see memory `vllm-zombie-enginecore-cleanup`): vLLM 0.19.1 leaves an EngineCore subprocess that `pkill` misses → kill by GPU uuid after EVERY eval:
```
for p in $(ssh wangys@192.168.12.43 'nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader|grep -E "96d51e17|12869e37"|cut -d, -f1'); do ssh wangys@192.168.12.43 "kill -9 $p"; done
```
(cuda6 uuid 96d51e17…, cuda7 uuid 12869e37…). Never kill a running training without asking (memory `never-kill-running-jobs-without-asking`).

---

## 2. REPRODUCTION — POLY (composable-MoE)
- Poly model trained via `revision_exp/poly_train.py` (PEFT PolyConfig, n_skills10/all-linear/2048/5ep). Adapters: `poly-mistral-7b`, `poly-qwen3-4b-v2` on 12.43.
- Bake Poly → per-task standard LoRA (vLLM-servable): `revision_exp/bake_poly_to_lora.py --poly poly-mistral-7b --out_root baked_lora_mi_7b` (verified numerically exact). Output `baked_lora_mi_7b/task-{AVE,CTA,DC,DI,ER,RE,SM}`.
- Eval (12.43 vllm_pascal, TP=2): `revision_exp/poly_vllm_eval.py --base <Mistral> --baked_root baked_lora_mi_7b --config poly_cfg_mi_vllm.json --tp 2`. Then metrics LOCALLY: `revision_exp/poly_eval_metrics.py --config poly_cfg_mi.json --pred_dir <csvs> --out_md ...`.
- **Format-normalize predictions before metrics**: `revision_exp/normalize_poly_preds.py <src_dir> <dst_dir>` (rescues `Output:"x"`/curly-quote/bare-label rows; never changes parseable rows). Mistral ER needed this (raw 0.35 → 0.79).

## 3. REPRODUCTION — DENSE (matched multi-task)
- Train on `union_n7` (31,166 ex = 7 DP tasks) via LLaMA-Factory on 51.11. Config `revision_exp/scaling_cfg/dense-n7-5ep.yaml` (Mistral-7B LoRA r16 all, mistral template, 5 epochs, **save_strategy: epoch** → ckpt-487/974/1461/1948/2435 = ep1-5). Launch: `FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train <yaml>`.
- Also: `dense-n7-10x.yaml` (0.5 epoch, max_steps 974) — the original under-trained baseline; epoch ckpts are the clean progression.
- Eval (12.43 TP=2): per-task DP via `revision_exp/dense_vllm_eval.py --config <cfg> --tp 2` (config = adapter path + 15 datasets); OOD via `revision_exp/ood_infer.py` (below). Metrics via `poly_eval_metrics.py` (DP) after `normalize_poly_preds.py`.

## 4. REPRODUCTION — OOD (MMLU / BBH) — uses StructInf's OWN metric, do NOT rewrite
- Eval data on 12.43: `ood/data/{mmlu,bbh}_eval.hf` (from `/home/yanmy/StructInf/example_LLM_pipeline/data/`).
- Generate: `revision_exp/ood_infer.py --base <model> --eval_hf ood/data/mmlu_eval.hf --out <pred.json> [--lora <adapter>] --max_new 16(mmlu)/256(bbh) --tp 2`. Saves `{prediction, output, response, subject}`.
- Metric: StructInf `offline_inference/eval_mmlu.py --input_path <pred.json>` and `eval_bbh.py` (copied to `ood/metric/` on 12.43). DO NOT hand-roll the scoring.
- base Mistral validates pipeline: MMLU 57.71 (≈published ~58%), BBH 36.43.

## 5. REPRODUCTION — ITS (intrinsic task subspace, AE-1/R1-1 geometric)
- Task vector = per-task qwen-0.5B LoRA (r16/alpha32/**q_proj**/template qwen/3ep, base Qwen2.5-0.5B-Instruct), via llama-factory.
- 16 DP adapters: `lora/qwen-0.5B/{TASK}/{ds}/init/adapter_model.safetensors` (local + 51.11).
- 3 non-tabular (grammar/mmlu/bbh): trained on 12.43 (`yaml_its_{grammar,mmlu,bbh}.yaml`, 900 examples each from `nontab_*.json`); registered via `data/dataset_info.json`; placed at `lora/qwen-0.5B/NONTAB/{name}/init/`.
- Run: `python revision_exp/expITS.py --glob 'lora/qwen-0.5B/*/*/init/adapter_model.safetensors' --out_dir revision_exp/results/ITS_v3` → `its.json` + `its_tsne.png` + `its_varexp.png`.
- grammar data = `datasets/grammars_train.hf` (synthetic string-transformation, 900). MMLU/BBH = StructInf-selected 5% subsample (900) from `revision_exp/ood_data/{mmlu,bbh}_structinf_select.json`.

## 6. WHERE ADAPTERS LIVE (large, gitignored — regenerate via §2-5)
- Poly baked: 12.43 `baked_lora_mi_7b/`, `baked_lora_q3_4b/`.
- dense epochs: 51.11 `revision_exp/lora/dense-n7-5ep/checkpoint-{487,974,1461,1948,2435}`; also copied to 12.43 `dense_adapters/dense-5ep-ep{1..5}/` and local `revision_exp/dense_adapters/`.
- MMLU/BBH experts (Mistral-7B): 51.11 `revision_exp/lora/{mmlu,bbh}-structinf-select/`.
- ITS 0.5B: local `lora/qwen-0.5B/` (+ NONTAB), 51.11 mirror.

## 7. OPEN / NEXT
- Fold R1-1 (OOD perf + ITS geometry aux) and AE-4 (parity reframe) into `MELDDS_TKDE_response_letter_REVISED.md` and the manuscript (blue edits). The response currently has older AE-1 "grammar" wording — relabel grammar as "synthetic string-transformation" and lead R1-1 with MMLU/BBH.
- Optional: random-5% vs StructInf-5% arm to make the selection-vs-random claim direct (only selected arm trained so far).
- Fold the MoE scalability/router experiments (§8) into the response (R1-2, R1-4/AE-5, R2-2.4) and manuscript (Figs A1/A2/B).

---

## 8. MoE SCALABILITY / ROUTER / MULTI-TASK COMPARISON (R1-2, R1-4/AE-5, AE-4, R2-2.4) — DONE 2026-06-18

Discriminative-only task suite (avoid DC/CTA: saturated/low-discriminative). 12 picks: ER{amazon-google,walmart-amazon,wdc,abt-buy,semi-text-w,semi-text-c}, DI{amazon,walmart}, SM{CMS,synthea}, RE{RE}, AVE{oa_mine}. Base = Mistral-7B. All scripts in `revision_exp/`; figs in `revision_exp/results/moe_figs/`.

### Exp A — router n-sweep (ONE sweep → scalability + stability + complexity)
`revision_exp/results/router_quality/router_quality_sweep.csv`; figs `figA1_scalability.png`, `figA2_stability_cost.png`. n=3..12, ×3 seed.
- **R1-4/AE-5 scalability**: routed task-F1 of the content router ≈ oracle (in-domain expert) at every n (gap ≤0.017) and ≫ random (+0.15–0.20) → routing quality does NOT degrade with n.
- **R1-2 stability**: expert utilization (normalized entropy) ~0.98 (no collapse); cross-seed F1 std shrinks as n grows (0.23→0.00).
- **R1-2 complexity**: router fit is seconds (LogisticRegression; a max_iter solver jump to ~13–15s at n≥10, still seconds), n_params linear in n, and the router NEVER retrains experts.
- Absolute F1 (~0.53–0.70) is dragged by the degenerate SM-CMS metric; the router-vs-oracle-vs-random comparison is the claim and is clean.

### Exp B — probe-F1 vs n: dense vs Poly+MHR vs MELD (all 1 epoch, unified)
`revision_exp/results/expB_probe/probe_f1_merged.csv`; fig `figB_probe_vs_n.png`. Fixed probe = {ER-amazon-google, DI-amazon, SM-CMS}; nested n=3/6/9/12. Headline metric = ER+DI mean (SM-CMS excluded, see caveat). dense = 2 seeds (s0,s1, tight std ≤0.003); poly = 1 seed; MELD = per-task experts (constant).
- **MELD (modular) ≈ 0.722, flat across n** (per-task experts do not interfere).
- **dense ≈ MELD (parity)**: 0.736 (n3) → ~0.710 (n6/n9/n12); starts ≥MELD, settles just below (within ~0.01). Supports **AE-4 parity**.
- **Poly+MHR (integrated MoE) is lowest at every n** (~0.67–0.70, single-seed noisy, non-monotonic) → supports **R2-2.4** (standalone-modular MELD more stable than integrated MoE).
- **CAVEAT — SM-CMS is a degenerate metric**: 36 positives / 5127 (0.7%), binary F1(pos=match) → all three methods ≈0 (even the dedicated MELD SM expert). Excluded from the headline; reported separately. ER/DI are the clean signals; DI is flat ~0.65–0.69 for all three (parity).
- Poly s1 (error bars) was attempted but dropped: 12.43 HF-Hub flakiness during poly_train (LocalEntryNotFoundError); poly stays single-seed. dense 2-seed error bars suffice to show the dense decline is real (not noise).

### Reproduction
1. `python revision_exp/expMoE_prep.py` → `manifest.csv`, dense unions `scaling_data/union-n{3,6,9,12}probe.json` (+register in LLaMA-Factory `dataset_info.json` with `dataset_dir`), poly per-task `poly_data/n{N}/task-*.json`.
2. **Exp A** (51.11, vllm 0.5.4): predict `expMoE_router_quality.py --phase predict --manifest manifest.csv --base <Mistral> --per_ds 300 --cache .../predmatrix`; route `--phase route --bge <bge> --ns 3..12 --seeds 0,1,2`. synthea expert via `scaling_cfg/synthea-expert.yaml` (note `dataset_dir`, output to `lora/mistral-7B/SM/synthea/select`).
3. **Exp B dense** (51.11): `dense_driver.sh <gpu> <n...>` (pre-gen `_dense_nN_s{0,1}.yaml`, llamafactory, 1ep). **Poly+MHR** (12.43, deepspeed env, sdpa): `poly_driver.sh <cuda6|7> <n...>` (`poly_train.py --n_skills 8 --n_splits 4 --epochs 1 --bsz 2 --cutoff 1024`). Eval (full test): `expMoE_eval_probe.py --mode {dense|poly|meld} --cap 6000` (dense=vLLM LoRA on 51.11; poly=DIRECT PEFT task_ids on 12.43, MHR can't bake; meld=per-task experts).
4. Figs: `make_moe_figs.py --sweep router_quality_sweep.csv --probe probe_f1_merged.csv` (Fig B uses ER+DI mean, excludes SM).

### GOTCHAS (cost real GPU-hours)
- `expMoE_eval_probe.py`/`expMoE_router_quality.py` carry `rescue()` + `er_sm_safe()`: Poly emits truncated dicts (`'mismatch'}`) and `Transfer()` scores any pred lacking "mismatch" as a positive match — both must be normalized (codex-reviewed; see memory `watchdog-and-codex-discipline`).
- Router route phase splits each dataset (eval=first half, train=second half) — first-half keeps RE's positional `test_RAG.csv` gold aligned; no train==test leakage.
- Multi-line `nohup bash -c '...'` over ssh silently fails / a bash for-loop with multiple ssh backgrounds only the first — launch ONE nohup per ssh or use a script file.
- `CUDA_VISIBLE_DEVICES=N` makes torch report the masked GPU as logical "GPU 0" — verify physical placement (uuid) before trusting; on 12.43 use cuda6/7 ONLY.
