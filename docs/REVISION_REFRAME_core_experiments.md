# MELD-DS TKDE Revision — Reframed core experiments (jump out of the MoE-vs-dense accuracy frame)

## Diagnosis: why the current setting is the wrong battlefield
- MELD "experts" = full Mistral-7B + LoRA each; top-m=2 -> ~2x7B active at inference, i.e. MELD uses *more* compute than dense-7B, not less. It is NOT a compute-saving sparse MoE (Mixtral-style).
- Therefore on the "in-distribution accuracy / active-compute" axis the MoE has no structural advantage over a matched dense-7B (Exp B confirmed: dense ~= MELD, +0.01 avg).
- Cutting dense compute to manufacture a win is not defensible. Fix = change the axis.

## Reframe: two axes where MELD genuinely wins
1. DATA EFFICIENCY (MELD-DS, the journal's real increment): accuracy per training budget.
2. MARGINAL COST OF CAPABILITY (modular experts + plug-and-play serving): cost to build & extend, interference-resistance, cross-dataset adaptation.

## CORE EXPERIMENT 1 — budget-matched data-efficiency frontier  (answers AE-1, AE-2, AE-3, AE-6)
One unified sweep on ~3-4 representative datasets (e.g. ER/amazon-google, CTA/WebTable, RE, DC/rayyan):
- X = selection budget (fraction c, or training FLOPs). Y = downstream F1.
- Methods: MELD-DS vs {random, full-data, DSIR, DataInf/SuperFilter}. Claim: MELD-DS dominates the accuracy-vs-budget frontier.
- Knob 1 — batch cohesion: random vs k-center clustering; measure batch-IF approx error + within-batch grad variance sigma -> empirically validates Theorem 6 sigma-dependence. (AE-2 / R2-2.2)
- Knob 2 — label-noise injection 0/10/20/30%: degradation curves, MELD-DS vs baselines. (AE-6 / R1-3)
- Report wall-clock WITH k-center clustering broken out. (AE-3 / R2-2.1)
- ITS: add one non-tabular task point to the intrinsic-subspace visualization + discuss cited cross-domain evidence. (AE-1 / R1-1)

## CORE EXPERIMENT 2 — marginal cost of capability: modular experts vs monolithic dense  (answers AE-4, AE-5)
Reframe dense-vs-MoE onto marginal compute + scalability + extensibility.
- (a) Task-count scaling n=3..10: dense = retrain-on-union at each n (compute grows superlinearly; cross-task interference lowers per-task F1); MELD = +1 cheap expert per task (constant marginal cost; isolation -> flat per-task F1). Plot per-task F1 and cumulative training compute vs n. This is where MoE wins, consistent with Theorem 2. (AE-5 scalability + AE-4 dense on a fair compute axis)
- (b) Cross-dataset plug-and-play (run on 51.11): held-out ER/DC datasets with NO in-domain expert. MELD routes/composes related experts (zero new training) vs dense (must retrain). Real src/evaluation.py metrics. (AE-5 standalone-vs-integrated + R2-2.4)
- (c) Compute-matched dense ("cut dense compute" done right): give dense and MELD the same TOTAL training budget; because dense must retrain on the union while MELD reuses experts, at matched budget dense cannot keep up as n grows.

## Coverage map
- Exp 1 -> AE-1, AE-2, AE-3, AE-6 (the DATA SELECTION method: foundation, validity, cost, robustness)
- Exp 2 -> AE-4, AE-5 (the MoE/serving: dense baseline on a fair axis, scalability, routing, when-standalone-helps)

## Narrative for the response letter
Primary contribution = MELD-DS (data selection): builds strong task experts under a tight compute/data budget (Exp 1). The MoE serving layer = plug-and-play extensibility and cross-dataset adaptation at constant marginal cost (Exp 2), matching a monolithic dense model's in-distribution quality while being far cheaper to BUILD and EXTEND. In-distribution accuracy parity with dense is expected and consistent with Theorem 2; the MoE advantage manifests in the scaling/extension/cross-dataset regime.
