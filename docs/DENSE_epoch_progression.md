# Dense (Mistral-7B LoRA on union_n7) — performance vs training epochs (15 DP + OOD, 100-scale)

CLEAN progression from the single `dense-n7-5ep` run (4-GPU, cosine over 5 epochs, save_strategy=epoch → ckpt-487/974/1461/1948/2435 = ep1/2/3/4/5). All eval on 12.43 dual-GPU vLLM (TP=2) + StructInf metric for OOD. Metrics: ER/DC/SM=F1, AVE/DI=Acc, CTA=micro/macro-F1, MMLU=5-shot, BBH=EM. base ref: MMLU 57.71 / BBH 36.43.

| Task | Dataset | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|---|
| ER | Amazon-Google | 66.38 | 79.14 | 79.16 | 79.63 | 80.69 |
| ER | Walmart-Amazon | 51.76 | 88.94 | 89.71 | 79.39 | 82.87 |
| ER | WDC-All | 77.95 | 91.47 | 91.92 | 90.32 | 91.55 |
| ER | Abt-Buy | 73.36 | 87.47 | 90.59 | 86.34 | 87.53 |
| ER | Semi-Text-Watch | 57.14 | 73.24 | 75.21 | 73.97 | 73.98 |
| ER | Semi-Text-Computer | 34.52 | 5.70 | 7.63 | 11.24 | 10.36 |
| DC | Hospital | 96.06 | 96.65 | 96.26 | 96.26 | 96.26 |
| DC | Rayyan | 81.97 | 85.07 | 85.45 | 84.98 | 84.94 |
| DC | Beer | 99.66 | 99.49 | 99.37 | 98.33 | 98.06 |
| CTA | WebTables micro/macro | 90.98/72.63 | 90.72/73.18 | 94.06/76.68 | 94.31/75.15 | 94.31/75.16 |
| DI | Amazon | 64.95 | 67.65 | 67.65 | 66.54 | 66.42 |
| DI | Walmart | 80.77 | 83.65 | 89.42 | 89.42 | 88.46 |
| AVE | OA-mine | 70.75 | 74.95 | 75.23 | 74.46 | 74.05 |
| SM | CMS | 15.30 | 14.63 | 8.70 | 23.29 | 29.06 |
| SM | Synthea | 13.33 | 14.29 | 24.24 | 22.99 | 28.17 |
| **OOD MMLU** | — | **57.99** | **55.13** | **55.75** | **55.48** | **55.30** |
| **OOD BBH** | — | **46.26** | **42.60** | **42.60** | **40.92** | **41.44** |

## FINAL summary (ep1→ep5, base ref MMLU 57.71 / BBH 36.43)
- **OOD forgetting (clean, monotone-then-plateau)**: MMLU 57.99→55.30 (−2.7, stabilizes ~2.4 below base); BBH 46.26→41.44 (−4.8, plateaus, **stays above base** thanks to cot-heavy union). The drop happens by ep2 then flattens → dense multi-task LoRA mildly but persistently erodes general OOD ability.
- **In-distribution DP rises and holds**: ER 66→80s/90s, CTA micro 91→94, DI walmart 81→88, SM recovers 15→29 by ep5. Dense specializes into DP as training grows.
- **The trade-off is the R1-1 evidence**: more dense training buys DP specialization at the cost of OOD general ability. MELD/Poly (frozen base + detachable LoRA) sidesteps this — OOD query → detach experts → base ability retained (Poly-ER even ≥ base: MMLU 58.16 / BBH 43.79). semi-text-computer is the known degenerate outlier (model stops emitting "match").

> Trend ep1→ep2: in-distribution DP climbs sharply (ER +13~37 pts) while OOD degrades (MMLU −2.9, BBH −3.7) = dense trades general ability for DP specialization as training grows. semi-text-computer is an outlier (model stops predicting "match": prec 0.94 / rec 0.03).

## ep1 observations (clean 5ep-schedule, ckpt-487)
- **OOD at ep1 ≈ base / better**: MMLU 57.99 ≈ base 57.71; BBH 46.26 ≫ base 36.43 (+9.8). At 1 epoch the dense LoRA has not yet overwritten general ability — and the cot-heavy union even helps BBH.
- **DP still ramping**: ER moderate (abt-buy 73, amazon-google 66, walmart-amazon 52), DC strong (beer 99.7), SM weak (OOD for dense). Expect ER to climb with more epochs; watch whether OOD then degrades (the forgetting hypothesis).

## Superseded reference (different runs/schedules — NOT comparable, kept for record)
| Task/Dataset | dense 0.5ep (dense-n7-10x) | dense "1ep" (dense-n7/ckpt-1948) |
|---|---|---|
| OOD MMLU | 55.33 | 51.55 |
| OOD BBH | 41.45 | 30.20 |
(These two were independent runs with different effective batch / LR schedules; the clean ep1 above is the authoritative 1-epoch point.)
