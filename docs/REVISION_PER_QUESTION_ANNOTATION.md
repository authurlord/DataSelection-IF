# Per-question annotation — TKDE-2025-09-3230 (问题 | 文章定义/位置 | 回答 | 数据支撑 | 状态)

Built on the response-to-reviewers draft; [TBD]s filled with real data; **AE-4/R2-2.3 corrected per actual results (dense ≈ MELD parity, NOT MELD-superior) — honest reframe**.
Status legend: ✅ closed · ⚠️ closed-with-honest-caveat · ✍️ writing-only (no experiment needed).

---

## AE-1 / R1-1 — ITS assumption beyond tabular  ✅
- **问题**: heterogeneous DP tasks share a common intrinsic subspace — insufficiently validated beyond tabular; test or discuss for structurally different domains.
- **文章定义/位置**: §III Theorem 1 (intrinsic task subspace) + Fig 2 (t-SNE of task vectors) + §II-B.
- **回答**: ITS is established across diverse NLP tasks in the literature (Aghajanyan 2021; Qin et al. — 100 heterogeneous tasks reparameterized in a few-hundred-dim subspace); we instantiate it for tabular-dense DP. Added a non-tabular (text-heavy relation/entity) task point to Fig 2; it forms a separable cluster within the SAME subspace. Limitations note added (free-text long-reasoning = future work).
- **数据支撑**: 16 tabular task-vectors → intrinsic dim 8 (90% var); adding a non-tabular task keeps dim 8, cosine 0.945 to the tabular cluster (co-embedded, not outlier).
- **viz**: `figs/expA_combined.png` (+ Fig 2 update needed in tex).

## AE-2 / R2-2.2 — Theorem 6 σ assumption under noisy/heterogeneous data  ✅ (forces a Thm-6 correction)
- **问题**: Thm 6 assumes low within-batch gradient variance σ from semantic similarity; may fail on noisy/heterogeneous data. Need empirical sensitivity under poor batch cohesion.
- **文章定义/位置**: §V-C Theorem 6 (batch-IF error ~ O(σ/(λ√|B|))); Table V (|B| sweep).
- **回答**: Added sensitivity study — (a) random vs k-center clustering, (b) inject cross-task samples to inflate σ; measure batch-IF approx error vs per-sample IF + downstream F1; extend Table V with measured σ. k-center keeps σ low; random clustering markedly worse → semantic clustering IS the mechanism. **Correction**: σ is NOT constant — σ(|B|) GROWS with |B|, so error is a σ(|B|)/√|B| tradeoff (not monotone decrease) — this explains the U-shaped sweet-spot in Table V.
- **数据支撑**: 3 datasets (RE/ER/CTA): error rises monotonically with σ; semantic clustering cuts σ AND error ~3.5× vs random; σ(|B|) increasing confirmed.
- **viz**: σ–error curve + |B|-U-curve (to generate).

## AE-3 / R2-2.1 — clustering overhead + novelty vs LESS/DataInf  ✅
- **问题**: 3–10% runtime claim ignores k-centered clustering cost; novelty vs LESS/DataInf.
- **文章定义/位置**: §V-C; Table III (runtime/init-time breakdown); §VII-C (related work).
- **回答**: Clustering reuses S'_init as centers (shares computation), greedy k-center O(nk), #clusters=|S'|/|B| ≪ n. Add clustering wall-clock to Table III. Novelty = integration of batch-IF with submodular diversity-quality (Eq 9) + small-proxy estimator under error bound (Thm 6/7) + multi-GPU, not batch-IF alone.
- **数据支撑**: clustering = **1.8–4.3% of selection time** (gradients 75–85%); e.g. WebTable 23.7s clustering / 570.9s total = 4.1%. 3–10% claim preserved.

## AE-4 / R2-2.3 — matched dense multi-task 7B + LLM-Baseline clarity  ⚠️ CORRECTED (parity, not superiority)
- **问题**: add a matched dense multi-task 7B to isolate MoE/router gain from confounds; clarify the "LLM Baseline" column.
- **文章定义/位置**: §VI-A; Table I (LLM Baseline column); Theorem 2 (MoE vs single-expert error bound).
- **回答 (CORRECTED — the draft's "MELD exceeds dense" is NOT supported; reframe to parity + modularity)**:
  Matched dense multi-task 7B (single Mistral-7B, all 10 tasks jointly, no MoE/router) added to Table I. **Honest finding: dense-7B reaches PARITY with MELD in static in-distribution accuracy** (consistent with Theorem 2, which separates the bounds only in low-capacity / low-N-per-task / large-n regimes — full-capacity 7B at small n is exactly the parity regime). MELD's value is therefore **NOT** an in-distribution accuracy amplifier; it is **modularity under parity**: per-task expert specialization, incremental onboarding without full-union retrain, and cross-dataset plug-and-play composition (R2-2.4). LLM-Baseline column clarified: per-task strongest offline LLM (JellyFish-13B for ER/DC/ED/DI; TableLLaMa-7B for CTA/RE/EL; ExtractGPT for AVE), distinct from the new matched dense 7B.
- **数据支撑 (real, my eval harness ≡ paper, verified RE micro-F1 0.770 = paper)**:
  - Exp B (dense 5-epoch, matched data): dense ≈ MELD, dense even +0.01 avg (parity).
  - Task-count scaling, mean per-task F1 (ER amazon-google + DC beer/hospital/rayyan): 10x-step dense (≈0.5ep) 0.860/0.868/0.865 vs MELD 0.892; dense-1ep 0.840/0.838/0.811. **No robust interference decay** — the 1-epoch dip is a training-stage transient, erased at 0.5ep and at 5ep. **Do NOT claim accuracy superiority.**
  - Per-task gap (MELD−10x-dense, n=7): ER +0.094, DC beer −0.002, hospital +0.003, rayyan +0.013 — discriminative ER favors MELD, saturated DC parity; but MELD used ~10× the training compute, so this is NOT an efficiency win either.
- **ACTION on tex**: rewrite the Table-I-discussion claim from "multiple sparse experts exceed the dense model (Thm 2)" → "comparable in-distribution accuracy; the MoE provides modular extensibility at parity"; soften abstract/§VI-B "superiority of MoE" wording.

## AE-5 / R1-2,R1-4,R2-2.4 — routing stability, task-count scalability, standalone-vs-integrated  ✅/⚠️
- **问题**: routing instability, scalability as #tasks grows, when standalone underperforms integrated MoE.
- **文章定义/位置**: §IV-C (Router Network); §VI-D/§VI-G; Table VI (cross-dataset), Table VII (ablation incl. Mixtral router).
- **回答**: (R1-2) Decoupled router optimizes a STATIONARY target (experts frozen first) → removes expert-router co-adaptation collapse (Chi 2022) / seed sensitivity; only cost = one contrastive pass. (R1-4) Task-count study n∈{3,5,7,10}: per-task F1, utilization, vs Mixtral. (R2-2.4) Standalone helps when tasks heterogeneous (different output domains: binary ER vs multiclass CTA vs generative DI); least advantageous for closely-related tasks differing only in output format (integrated MoE could share early layers, but with joint-training instability).
- **数据支撑**: router stable 3→18 experts (top-1 1.0→0.951, top-2 ≥0.976, balanced util); cross-dataset top-m composition robust — top-m ≈ best-single, > mean-single (+0.085 avg), >> worst-single (abt-buy +0.43, walmart +0.41); honest: top-m does NOT beat best-single (robustness, not accuracy). dense per-task F1 ~flat/parity with n (no clear interference) — frame scalability as **operational** (constant marginal cost, see compute below), not accuracy.
- **viz**: `figs/router_scalability.png`, `figs/moe_crossdataset_topm.png`, `figs/moe_crossds_ER_DC.png`.

## AE-6 / R1-3 — noisy/biased self-annotated augmentation robustness  ✅
- **问题**: self-annotated/augmented data may add bias/noise; need robustness evidence.
- **文章定义/位置**: §IV-A (augmentation), §V (selection); §VI-[C] (new), Fig [Y].
- **回答**: Inject 0/10/20/30% label noise; compare MELD-DS vs full-pool / DSIR / SuperFilter on Amazon-Google, WebTable, Rayyan. PPL-ratio (quality) + batch-IF (complexity) filter injected noise → MELD-DS degrades slower.
- **数据支撑**: ppl-filtering captures **94.3% of injected noise** (91.1% of dropped samples are the injected noise); MELD-DS degrades ~2× slower than random-30%. (Lead with the filtering result; training-degradation has single-seed variance.)

---

## R2-3.1 ✍️ Fig-1 EM→DI transform — fully automated (schema-level mask operator, no per-instance rules). §II-B/§IV-A.
## R2-3.2 ✍️ Eq-1 mutual information — conceptual objective, operationalized by PEFT label-likelihood (inner) + controlled RAG data (outer); no MI estimator. §IV-C after Eq 1.
## R2-3.3 ✍️ PPL-ratio thresholding — conservative heuristic; ratio>1 mostly = self-annotation label noise, not informative hard negative (hard negatives help only with a contrastive objective, Yan 2024). §V-B.
## R2-3.4 ✅ Init-time 1200s(DataInf) vs 115s(MELD) — DataInf precomputes per-sample gradients of full 7B over pool (22GB); MELD warms 0.5B proxy + batch-level gradients (3.4GB). No IF step omitted; reduction = proxy-scale + batch-level. §V-C Table III.
## R1-5 ✍️ Failure cases — MELD weaker on open-domain retrieval-heavy (AVE/oa_mine, CTA macro-F1/SemTab19 where Mixtral's 8×7B has broader knowledge); strongest on closed-domain structural tasks. Appendix table.

---

## NEW (not in draft) — robust MoE-justification axes for AE-4/AE-5 (compute, since accuracy is parity)
- **Selection compute** (answers AE-3 + strengthens the data-selection contribution): MELD-DS selection = 13.4 PFLOP (0.5B proxy, batch-level IF) vs **LESS 424.7 PFLOP (32× cheaper)**, QuRating 25.3 PFLOP (1.9×).
- **Build/extension compute**: building 7 tasks one-at-a-time, dense retrains the growing union each time = **5.2× cumulative train FLOP** vs MELD (+1 expert each, linear). Frame strictly as continual-onboarding (concede: all-tasks-upfront → dense same cost).
- **Inference**: state plainly MELD top-m ≈ **2× dense** (each expert a full 7B) — a cost traded for modularity.
