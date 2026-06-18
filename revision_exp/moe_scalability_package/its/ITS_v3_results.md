# ITS (intrinsic task subspace) — reproducible 19-task analysis (AE-1 / R1-1)

All task vectors = per-task qwen-0.5B LoRA adapters (r16 / alpha32 / **q_proj** / template qwen / 3 epochs, base Qwen2.5-0.5B-Instruct), via llama-factory. The 3 non-tabular points were retrained by us with this documented process (provenance controlled), 900 examples each (matched to grammar's 900). Figure: `revision_exp/results/ITS_v3/its_tsne.png`, var-curve `its_varexp.png`, data `its.json`.

## Result
- **19 tasks** = 16 tabular/semi-structured DP + 3 structurally-different non-tabular (NONTAB-grammar = synthetic sentence string-transformation; NONTAB-mmlu = general-domain QA; NONTAB-bbh = multi-step reasoning).
- Parameter dim 688,128, but **intrinsic dimensionality 90%=10 / 95%=13** (16-DP-only was 8). Adding 3 genuinely different general-domain tasks raises it only slightly → the shared low-dimensional subspace persists.
- Mean off-diagonal cosine 0.969. Non-tabular points co-embed (high cosine to the DP cluster, not outliers):
  - NONTAB-grammar: mean cos to DP = **0.945** (reproduces the earlier value exactly)
  - NONTAB-mmlu: **0.967**
  - NONTAB-bbh: **0.966**

## Interpretation (for R1-1 / AE-1)
The intrinsic-task-subspace property is NOT confined to tabular DP: even general-domain QA (MMLU), multi-step reasoning (BBH), and synthetic string transformation (grammar) have task vectors that compress into the same ~10-dimensional shared subspace and sit close to the tabular DP cluster. This is the geometric evidence that the ITS assumption "holds for structurally different domains". It is a GEOMETRIC auxiliary result; the primary OOD evidence for R1-1 is the MMLU/BBH performance study (see DENSE_epoch_progression.md / REVISION_OOD_R1-1_results.md).

## Provenance note (honesty)
The earlier its_v2 NONTAB-grammar point was verified legitimate: trained on 900 grammar examples (its_grammar.log: "Num examples = 900"), same 0.5B LoRA process. The original yaml's `dataset: all_tasks` field was a misleading label; `train_file_path: nontab_grammar.json` is what actually loaded (900 = grammar size confirms). The v3 run here retrains all 3 non-tabular points cleanly to remove any provenance ambiguity.
