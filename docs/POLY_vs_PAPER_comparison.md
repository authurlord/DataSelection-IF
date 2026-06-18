# Paper MELD (Table I) vs Poly composable-MoE — per-task comparison (100-scale)

- **论文 MELD**: full system = per-task LoRA experts + RAG + standalone router, trained on full augmented data (Table I, "MELD Few-shot" column).
- **Poly**: composable-by-construction MoE = ONE shared multi-skill PEFT-Poly model, routed per task_id and baked to per-task standard LoRA. Single model covers all tasks (vs paper's separate expert per task).
- Metrics: ER/DC/SM = F1; AVE/DI = Accuracy; CTA = micro/macro-F1. Poly scores ×100, faithful format-normalized.

| Task | Dataset | Paper MELD | Poly Mistral-7B | Poly Qwen3-4B |
|---|---|---|---|---|
| EM/ER | Amazon-Google | 83.41 | 75.56 | 76.46 |
| EM/ER | Walmart-Amazon | 91.42 | 82.90 | 79.48 |
| EM/ER | WDC-All | 91.97 | 87.05 | 90.23 |
| EM/ER | Abt-Buy | 91.12 | 79.00 | 85.84 |
| EM/ER | Semi-Text-Watch | 78.28 | 73.16 | 76.65 |
| EM/ER | Semi-Text-Computer | 86.46 | 25.12 ⚠️ | 24.44 ⚠️ |
| DC | Hospital | 95.01 | 95.06 | 94.67 |
| DC | Rayyan | 82.15 | 84.77 | 83.55 |
| DC | Beer | 97.30 | 99.85 | 96.68 |
| CTA | WebTables (micro/macro) | 96.41 / 78.76 | 92.67 / 73.31 | 91.48 / 74.52 |
| DI | Amazon | 75.12 | 66.30 | 65.20 |
| DI | Walmart | 87.50 | 84.62 | 85.58 |
| AVE | OA-mine | 74.62 | 72.91 | 70.50 |
| SM | CMS | 60.27 | 34.38 | 25.71 |
| SM | Synthea | 56.00 | 27.16 | 13.11 |

## Observations
- **DC: Poly ≈ or BEATS paper** (Beer 99.85>97.30, Rayyan 84.77>82.15, Hospital ≈) — the shared composable model matches dedicated experts on data-cleaning.
- **ER: Poly within ~5–12 pts of paper on most** (WDC 87–90 vs 92, Abt-Buy q3 85.8 vs 91, Semi-Text-Watch ≈); paper's dedicated per-task experts retain an edge. **Semi-Text-Computer is a genuine failure** for Poly (~25 vs 86): the composable model does not emit a label on this dataset (~87% unparsable, reproduces description text) — a real OOD-style breakdown, not a format artifact.
- **DI / AVE: Poly slightly below paper** (within ~3–9 pts).
- **SM (schema matching): Poly well below paper** (paper 60/56 vs Poly 34/27 mistral, 26/13 q3) — hardest task for a single shared model.

## Takeaway (for the MoE serving discussion)
The composable Poly MoE — a SINGLE shared multi-skill model covering all tasks, vs the paper's separate expert-per-task — recovers most of MELD's accuracy on DC (parity/better), DI, AVE, and easy/medium ER, at the cost of a notable drop on the hardest tasks (Semi-Text-Computer, SM). This supports framing Poly/composition as a modularity-and-deployment convenience that trades some peak accuracy on the hardest tasks, not an accuracy win over dedicated experts.
