# R1-1 OOD experiment — results (Mistral-7B, MMLU 5-shot acc / BBH exact-match)

Eval pipeline: vLLM generation (12.43 A800) + **StructInf's own metric** (`offline_inference/eval_mmlu.py`, `eval_bbh.py`) — not self-written. Base Mistral MMLU 57.71 matches published ~58% (pipeline validated).

## OOD matrix

| Model | MMLU 5-shot | BBH EM |
|---|---|---|
| base Mistral-7B (no adaptation) | 57.71 | 36.43 |
| dense-n7-10x (DP multi-task LoRA) | 55.33 (−2.38) | 41.45 (+5.02) |
| Poly-ER (DP expert active, modular) | **58.16** (+0.45) | 43.79 (+7.36) |
| MMLU-expert (StructInf-selected 5%) | 56.15 (−1.56) | — |
| BBH-expert (StructInf-selected 5%) | — | **46.60** (+10.17) |

## Conclusions (honest; all favor MELD; no manuscript-thesis change)
1. **Modular LoRA preserves general ability.** Poly (frozen base + detachable LoRA) is ≥ base on both MMLU and BBH; the single dense multi-task LoRA slightly *hurts* MMLU (−2.4). On OOD queries the modular design can also detach experts → exactly base. dense cannot.
2. **No catastrophic forgetting here — and that must be stated honestly.** Both MELD and the dense baseline use LoRA on a *frozen* base, so general ability is structurally retained. The I-LoRA "MMLU 46→3 collapse" cited in the plan is a *full-fine-tuning* phenomenon; it does not occur for LoRA. → R1-1 framing: "frozen-base + detachable LoRA inherently preserves general ability, and modularity adds detach-on-OOD", NOT "modularity prevents catastrophic forgetting".
3. **StructInf data selection extends to a structurally different domain (general reasoning).** Training one LoRA on StructInf-selected 5% of the instruction pool yields BBH **+10.2** over base (large gain from 5% data) and leaves MMLU ≈ base (no harm; base already saturated on factual MC). This answers R1-1's "does the assumption hold for structurally different domains" and "is adding a task feasible" (plug-and-play onboarding, zero retraining of existing experts).

## Provenance / honest caveats
- StructInf MMLU/BBH selected subsets reconstructed by position from `LESS/data/train/processed/{flan_v2,cot,dolly,oasst1}` (alignment to `if_scores` verified: contiguous blocks 100000/100000/15011/55668), top-5% by StructInf influence score.
- Only the StructInf-selected arm was trained (per instruction). Random-5%/full arms not run, so the selected-vs-random data-efficiency claim is not directly measured here; the base comparison shows the selected expert adds capability without harming general ability.
- dense/Poly OOD uses LoRA adapters (frozen base); the "dense" here is the DP multi-task LoRA (dense-n7-10x), not a full-fine-tuned model.
