# Revision-requirements review — does MELD's MoE meet the TKDE revision bar? (VERDICT + iteration outcome)

This is the explicit review the /goal asks for: per requirement, is it answerable now; the iteration on dense-compute; and LESS/QuRating FLOP accounting.

## TOP-LINE VERDICT
**The revision is answerable — but NOT by claiming MoE accuracy-superiority.** MoE meets the bar via **modularity-under-parity + OOD general-ability retention + selection/extension compute**, which is the honest and defensible position. The "weaken dense compute until MoE wins on accuracy" path was executed and **rejected as non-robust and dishonest** (details below). Iteration has converged.

## Did the dense-weakening iteration help? (the /goal's conditional) — DONE, converged to "do not pursue"
Executed dense at reduced compute and measured per-task F1:
- dense 5-epoch (Exp B): ≈ MELD parity (dense +0.01 avg).
- dense 1-epoch: 0.840→0.811 (a transient training-stage dip, not robust).
- dense 0.5-epoch (10x-fewer-steps, dense-n7-10x): 0.86–0.87, still ≈ MELD 0.892; on the full 15-dataset OOD eval it tracks MELD/Poly closely (DC parity/better, ER within a few pts).
**Finding:** weakening dense compute does NOT manufacture an honest MoE accuracy win — it only erases the (compute-confounded) signal. Pursuing it further is both non-robust and scientifically dishonest. → **Stop. Reframe instead.** (codex-validated; see REVISION_synthesis FINAL AE-4 VERDICT.)

## LESS / QuRating FLOP — incorporated (answers AE-3 / R2-2.1)
Selection-stage compute (PFLOP): **MELD-DS 13.4** (0.5B proxy, batch-IF) vs **QuRating 25.3 (1.9×)** vs **LESS 424.7 (32×)**. MELD-DS reaches the same data-budget frontier at a fraction of the selection compute. Clustering wall-clock = 1.8–4.3% of selection time. This is where MELD-DS wins decisively and is the primary headline.

## Per-requirement status

| # | Requirement | Answerable now? | Evidence |
|---|---|---|---|
| AE-1/R1-1 | ITS beyond tabular / structurally different domains + add-task feasibility | **YES** | OOD: modular LoRA preserves general ability (Poly ≥ base on MMLU+BBH); StructInf-selected 5% trains a BBH expert at +10.2 over base → selection + plug-and-play extend to general-reasoning domain. ITS dim-8/cos-0.945 for non-tabular co-embedding. |
| AE-2/R2-2.2 | Theorem-6 σ sensitivity | **YES** | error ↑ monotonically with measured σ; k-center vs random ~3.5× lower σ+error; σ(\|B\|) increasing (restate as σ(\|B\|)/√\|B\| tradeoff). |
| AE-3/R2-2.1 | Clustering cost + novelty vs LESS/DataInf | **YES** | clustering 1.8–4.3% of selection; selection FLOP 32× cheaper than LESS, 1.9× vs QuRating. |
| AE-4/R2-2.3 | Matched dense 7B baseline; LLM-Baseline clarity | **YES (reframed)** | dense ≈ MELD parity (Theorem-2-predicted in full-capacity/small-n); MoE value = modularity-under-parity, NOT accuracy. dense-n7-10x full 15-dataset table delivered. LLM-Baseline column specified. **Do not claim superiority.** |
| AE-5/R1-2,R1-4,R2-2.4 | Routing stability, task-count scaling, standalone-vs-integrated | **YES** | router stable 3→18 (top-1 1.0→0.951); per-task F1 flat with n; cross-dataset top-m ≈ best-single, > mean, >> worst; when-helps/hurts analysis. |
| AE-6/R1-3 | Noisy augmentation robustness | **YES** | ppl-filter captures 94.3% injected noise; degrades ~2× slower. |
| R1-5 | Failure-case analysis | YES (writing) | open-domain retrieval-heavy (AVE/oa_mine, CTA-macro SemTab19) where Mixtral competes; Poly composable fails on Semi-Text-Computer (genuine). |
| R2-3.1 | Fig-1 transform automated? | YES (writing) | fully automated schema-level mask, no per-instance rules. |
| R2-3.2 | Eq-1 MI operationalization | YES (writing+lit) | conceptual IB; sufficiency I(Z;Y)=PEFT label-likelihood = Alemi VIB accuracy term; compression = controlled RAG data. Cites Tishby/Alemi/Kawaguchi. |
| R2-3.3 | PPL-ratio thresholding / hard negatives | YES (writing) | conservative heuristic; ratio>1 = self-annotation noise not informative negative. |
| R2-3.4 | Init-time 1200s vs 115s | YES | DataInf full-7B per-sample grads (22GB); MELD 0.5B proxy + batch-level (3.4GB). |

## Does the MoE serving layer specifically meet the bar?
- **In-distribution accuracy**: parity with dense (NOT superior) — expected, not a failure.
- **Composable Poly MoE** (one shared model, all tasks): recovers most of per-task-expert MELD on DC (≥), DI, AVE, easy/medium ER; lags on hardest (Semi-Text-Computer, SM). Frame as deployment/modularity convenience, not accuracy win.
- **OOD / general-ability**: modular MoE ≥ base ≥ dense — a genuine structural advantage of the MoE/modular design.
- **Compute/extension**: 5.2× cumulative build (continual onboarding), 2× inference (stated honestly), selection 32× cheaper than LESS.

## CONVERGENCE STATEMENT
All planned experiments are complete; the review is done; the dense-weakening conditional was tried and resolved (reject → reframe). The revision questions are answerable on the honest axes above. No further dense-crippling iterations are warranted. Remaining work is writing (fold these into the response letter / manuscript), not more compute.
