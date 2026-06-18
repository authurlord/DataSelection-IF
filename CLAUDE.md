# CLAUDE.md — DataSelection-IF (MELD TKDE revision)

This repo holds the MELD / MELD-DS work (data selection + MoE serving) and its TKDE R1 revision experiments.

## START HERE for the revision
Read **`docs/HANDOFF.md`** — it is the single source of truth: current state, all result docs, and exact reproduction commands for Poly, dense, OOD (MMLU/BBH), and ITS. The revision tracking lives under `docs/` and `revision_exp/`.

## Machines (only these; never grab others' GPUs)
- **51.11** (yanmengyi): 4×V100-32G, all usable. llama-factory training; conda `deepspeed`; repo `/data/yanmengyi/DataSelection-IF`, LF at `/data/yanmengyi/LLaMA-Factory`.
- **12.43** (wangys): A800-80G, **cuda6/7 ONLY**, don't preempt others. vLLM eval env `vllm_pascal` (0.19.1); llama-factory in `wangyaoshu/.../deepspeed`; `/data/home/wangys/meld_revision`.
- **51.10** = LOCAL (this repo): RTX 3090 **cuda2 only**, release immediately. base env torch2.11/tf4.57/peft0.18.

## Hard rules (cost the user real GPU-hours / trust when violated)
1. **After EVERY vLLM eval on 12.43**: kill leftover EngineCore by GPU uuid (pkill misses it) and verify mem≈0. See memory `vllm-zombie-enginecore-cleanup`.
2. **NEVER kill/restart a running training without asking** — surface the GPU-h cost first. See memory `never-kill-running-jobs-without-asking`.
3. **Use existing tools/scripts, don't hand-roll**: training = llama-factory; OOD metric = StructInf `eval_mmlu.py`/`eval_bbh.py`; eval = `revision_exp/{poly_vllm_eval,dense_vllm_eval,ood_infer}.py`.
4. Confirm cuda6/7 free before launching; dual-GPU vLLM = `--tp 2` on clean GPUs.

## Honest conclusions to preserve (revision)
- AE-4: matched dense ≈ MELD **parity** in-distribution; reframe to modularity-under-parity. Do NOT claim MoE accuracy superiority; do NOT cripple dense.
- R1-1: OOD (MMLU/BBH) is the primary evidence — dense forgets with training (MMLU 58→55, BBH 46→41 over ep1-5), modular MELD/Poly retains base (Poly-ER ≥ base). ITS geometry (dim 10, grammar/mmlu/bbh co-embed) is auxiliary.
- MELD-DS selection 32× cheaper than LESS.

## Eval format note
Poly/dense predictions need `revision_exp/normalize_poly_preds.py` before `poly_eval_metrics.py` (rescues loosely-formatted `Output:"x"` rows; never alters parseable ones).
