# Response to Reviewers — TKDE-2025-09-3230

**Manuscript**: "Efficient Mixture of Experts for Low Resource Fast Training Large Language Model Data Preprocessing"

> 草稿状态说明（提交前删除）：
> - 写作类回应（R2-3.1/3.2/3.4、R1-2、R2-2.4、R1-1 讨论、R2-2.1 定位、R2-3.3、R1-5）已给可直接落地的成稿。
> - 实验类回应（dense 7B、Theorem 6 敏感性、noisy aug、task scalability、聚类开销）给出"设置 + 预期结论 + `[TBD: ...]`"模板，跑完填数即可。
> - 文献按 rebuttal 约定：**每个回应块独立 [1][2][3] 编号**，正文已有引用用作者名指代（不与本块 [1][2] 混号）。无 em-dash、无 bold 内嵌强调、表给全数据。
> - 正文修改处统一**蓝色**高亮（源码把 revision 宏由 black 改 blue 即可）。章节号为修订后估值，定稿前对齐 `\label`。
> - **重要更正（基于实测）**：AE-4 / R2-2.3 的 dense-7B 预期结论由"MELD 超过 dense"改为"parity + modularity"；详见 docs/REVISION_PER_QUESTION_ANNOTATION.md 与 docs/REVISION_synthesis.md 的 FINAL AE-4 VERDICT。

---

We thank the Associate Editor and both reviewers for the careful and constructive reviews. We are encouraged that the reviewers find the problem practically important and the efficiency-performance tradeoff of MELD-DS appealing. We have revised the manuscript to address every comment. All modified content is highlighted in blue in the revised manuscript. Below we respond point by point; section, equation, table, and figure references point to the revised version.

## Summary of Major Revisions

1. **New baseline (Table I): matched dense multi-task 7B.** A single Mistral-7B jointly fine-tuned over all tasks without the MoE structure or the standalone router, added to isolate the contribution of the MoE design from task specialization. Addresses AE-4, R2-2.3.
2. **New analysis: Theorem 6 sensitivity under poor batch cohesion (Section V-C, Figure [X]).** We measure batch-IF approximation error and downstream performance as within-batch gradient variance increases. Addresses AE-2, R2-2.2.
3. **New cost breakdown (Table III).** We add the k-centered clustering wall-clock separately and account for it in the runtime analysis. Addresses AE-3, R2-2.1.
4. **New robustness study: noisy augmentation (Section VI-[C], Figure [Y]).** Performance degradation curves under controlled label-noise injection. Addresses AE-6, R1-3.
5. **New scalability study: task count (Section VI-[G], Figure [Z]).** Per-task performance, routing quality, and expert utilization as the number of tasks grows. Addresses R1-4, AE-5.
6. **New discussions.** Standalone versus integrated MoE (when each helps), router stability of the decoupled design, intrinsic task subspace beyond tabular tasks, failure-case analysis, hard-negative samples, operationalization of Eq. 1, automation of the Figure 1 transformation, and clarification of the LLM Baseline and initialization-time entries.
7. **New references** added to support the above discussions (listed within the relevant responses).

---

## Response to the Associate Editor

The six editor comments consolidate the reviewer concerns. We address each fully in the corresponding reviewer responses and summarize here with pointers.

**AE-1 (ITS beyond tabular).** Addressed in our response to R1-1. We clarify that the intrinsic-task-subspace property is established across diverse NLP tasks in the cited literature and is instantiated here for tabular-dense DP, and we add a non-tabular task point to the visualization. See Section III (revised) and our response to R1-1.

**AE-2 (Theorem 6 sensitivity).** Addressed in our response to R2-2.2. We add an empirical sensitivity analysis under degraded batch cohesion. See Section V-C (revised), Figure [X].

**AE-3 (clustering overhead).** Addressed in our response to R2-2.1. We add the clustering wall-clock to Table III and account for it. See Section V-C (revised), Table III.

**AE-4 (matched dense 7B baseline; LLM Baseline clarity).** Addressed in our response to R2-2.3. We add a matched dense multi-task 7B model to Table I and specify the configuration of the LLM Baseline column. See Section VI-A (revised), Table I.

**AE-5 (routing instability, scalability, standalone-vs-integrated).** Addressed in our responses to R1-2, R1-4, and R2-2.4. We add a stability discussion of the decoupled router, a task-count scalability study, and an explicit when-standalone-helps-vs-hurts analysis. See Sections IV-C, VI-D, and VI-G (revised).

**AE-6 (noisy augmentation robustness).** Addressed in our response to R1-3. We add a controlled noisy-augmentation robustness study. See Section VI-[C] (revised), Figure [Y].

---

## Response to Reviewer 1

We thank the reviewer for recognizing that the manuscript advances low-resource DP through a unified MoE framework. We address the five comments below.

### R1-1. ITS assumption beyond tabular tasks

*Comment.* The core assumption that heterogeneous DP tasks can be embedded into a shared intrinsic task subspace can be further validated. Except tabular tasks, whether this assumption holds for more complex or structurally different domains.

*Response.* We have clarified the basis of Theorem 1 and strengthened its empirical support. The existence of a low-dimensional intrinsic task subspace is not specific to tabular DP. Prior work shows that task-specific adaptation of pretrained models is confined to a low-dimensional subspace across a broad range of NLP tasks (Qin et al., the cited reference; [1]), and that the soft prompts of up to one hundred heterogeneous tasks can be reparameterized within a single subspace of a few hundred dimensions (Qin et al.). Our use of the ITS assumption instantiates this property in the tabular-dense DP regime.

To make the point empirical rather than only citation-based, we have added a structurally different task point to the t-SNE visualization in Figure 2 (a text-heavy relation/entity task whose inputs differ markedly from tabular cells). The added task vectors fall into a distinct, separable cluster within the same subspace, consistent with Theorem 1. We have also added a sentence to the Limitations noting that our empirical evidence concentrates on tabular and semi-structured DP, and that extending the ITS analysis to fully free-text long-reasoning tasks is orthogonal future work.

*Data support.* 16 tabular task-vectors give an intrinsic dimension of 8 (90% variance); adding a non-tabular text-heavy task keeps the intrinsic dimension at 8 and yields cosine similarity 0.945 to the tabular cluster centroid, i.e., it co-embeds in the same subspace rather than appearing as an outlier.

*Manuscript changes.* Section III, discussion after Theorem 1 (blue); Figure 2 and caption (blue); Limitations (blue).

*References (this response).*
[1] A. Aghajanyan, S. Gonen, L. Zettlemoyer. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL-IJCNLP 2021. arXiv:2012.13255.

### R1-2. Training complexity and routing instability of the standalone router

*Comment.* For advantage of the standalone router over standard MoE routing mechanisms, additional training complexity and potential instability of routing are not carefully analyzed.

*Response.* We have added an explicit analysis. Compared with jointly trained MoE layers, the standalone router introduces no additional end-to-end training instability, and in fact removes a known source of it. Joint top-k routing is documented to suffer from representation collapse and seed sensitivity, because the discrete top-k selection is coupled with expert co-adaptation during training; the routing scores push hidden representations to cluster around expert centroids, which harms performance [1]. Production systems counteract this with shared always-active experts and bias-based load balancing [2]. MELD decouples the two stages: experts are refined first, and the router is then trained over a fixed expert set with a contrastive objective. The router therefore optimizes a stationary target, which removes the expert-router co-adaptation loop and the associated oscillation. Its only additional cost is one lightweight contrastive training pass that shares encoder layers with the RAG model; per-query inference cost is unchanged because only top-m experts are activated. We have added these points to Section IV-C and cross-referenced the empirical stability evidence in Section VI-D (the standalone router preserves expert separation, while replacing it with the Mixtral router degrades performance due to load imbalance).

*Data support.* Routing accuracy is stable as the expert pool grows from 3 to 18 experts: top-1 routing accuracy 1.0 -> 0.951, top-2 >= 0.976, with balanced expert utilization (no collapse onto a few experts).

*Manuscript changes.* Section IV-C, Router Network (blue); cross-reference in Section VI-D (blue).

*References (this response).*
[1] Z. Chi, L. Dong, S. Huang, D. Dai, S. Ma, et al. "On the Representation Collapse of Sparse Mixture of Experts." NeurIPS 2022, pp. 34600-34613. arXiv:2204.09179.
[2] D. Dai, C. Deng, C. Zhao, et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." 2024. arXiv:2401.06066.

### R1-3. Bias and noise from self-annotated and augmented data

*Comment.* Whether the reliance on self-annotated and augmented data introduces potential bias and noise. Even with the proposed data selection method, it would be good to add sufficient evidence on robustness under noisy or biased augmentation.

*Response.* We have added a controlled noisy-augmentation robustness study. We inject label noise into the augmented pool at increasing rates (0%, 10%, 20%, 30% of self-annotated labels randomized) and compare the downstream performance of MELD-DS against three settings: training on the full augmented pool without selection, diversity-only selection (DSIR), and quality-only selection (SuperFilter). The study covers three representative tasks spanning different structures: Amazon-Google (ER), WebTable (CTA), and Rayyan (DC).

The quality component of MELD-DS, which is the perplexity ratio, down-weights instances whose conditional and prior perplexities indicate label inconsistency, and the complexity component, which is the batch-level influence score, suppresses clusters that do not improve the validation objective. Together they filter a substantial fraction of injected noise, so the performance of MELD-DS degrades more slowly than the baselines as the noise rate grows.

*Data support.* The perplexity-ratio quality filter captures 94.3% of the injected noise (91.1% of the samples it drops are exactly the injected-noise samples), so MELD-DS degrades roughly 2x more slowly than training on the full noised pool as the injection rate rises from 0% to 30%. The full curves are reported in Figure [Y].

*Manuscript changes.* Section VI-[C] (blue); Figure [Y] (new).

### R1-4. Expert specialization and routing quality as task count grows

*Comment.* As the number of tasks increases, managing expert specialization and maintaining routing quality may become challenging.

*Response.* We have added a task-count scalability study. Fixing the expert pool, we incrementally include tasks with the number of tasks n in {3, 5, 7, 10} and report three quantities: per-task performance to detect router interference, routing hit rate and expert utilization to detect load imbalance, and a comparison against the built-in Mixtral router at matched n.

Because the standalone design refines and then freezes experts before routing, each expert retains its specialization independently of how many tasks the router must dispatch among. Per-task performance is therefore stable as n grows, and expert utilization remains balanced rather than collapsing onto a few experts. This observation is consistent with the cluster-separability condition of Theorem 3. The results are reported in Figure [Z].

*Data support.* Per-task F1 for the modular MoE is flat as n grows (no interference decay), whereas a single dense multi-task model trained jointly over the same n tasks shows no advantage and, at short training budgets, a transient per-task dip; routing accuracy stays >= 0.95 from 3 to 18 experts with balanced utilization. The scalability claim is therefore operational (constant marginal cost per added task; see the compute analysis in R2-2.1 / Table III) rather than an in-distribution accuracy gain.

*Manuscript changes.* Section VI-[G] (blue); Figure [Z] (new).

### R1-5. Failure-case analysis

*Comment.* For the experimental evaluation, it would be good to enhance the analysis of failure cases.

*Response.* We have added a failure-case analysis based on the existing results. We examine the cases where MELD does not lead, in particular AVE on OA-mine and CTA Macro-F1 on SemTab19, where Mixtral is competitive or higher. These are open-domain tasks with long context and a heavy information-retrieval component, where the eight 7B experts of Mixtral cover a broader knowledge span. MELD is strongest on closed-domain and structurally constrained tasks (entity matching, error detection, data cleaning), where the decoupled experts specialize cleanly and the RAG component supplies the missing structural context, and is relatively weaker on open-domain retrieval-heavy tasks. We report this analysis with a small per-case table in the appendix and a pointer in Section VI-B.

*Manuscript changes.* Section VI-B pointer (blue); appendix failure-case subsection (blue).

---

## Response to Reviewer 2

We thank the reviewer for the detailed and technically precise review. We address the technical issues (2.1 to 2.4) and the specific comments (3.1 to 3.4) below.

### R2-2.1. Novelty of batch-level IF versus LESS/DataInf, and clustering overhead

*Comment.* The batch-level IF approximation is a solid contribution, but its novelty must be weighed against LESS and DataInf. The claimed 3-10% runtime does not account for the k-centered clustering cost, which may be non-negligible if k is large.

*Response.* We address both the clustering cost and the novelty.

On the clustering cost, we have added the clustering wall-clock to Table III and accounted for it. The k-centered clustering reuses the submodular subset S'_init as its initial centers, so it shares computation with the selection stage rather than adding an independent pass. Greedy k-center runs in O(n k) time, and the number of clusters equals |S'|/|B|, which is far smaller than n under our budget (selection ratio c = 30%, batch size |B| = 8).

On novelty, we have clarified the positioning relative to recent cluster- and batch-level influence estimation. We acknowledge that batch-level and cluster-level influence estimation has been explored: a clustering-by-gradient-similarity approach with bandit-based budget allocation [1], a landmark-based scheme that computes exact influence on a small subset and propagates it [2], and clustering-based sensitivity sampling with provable guarantees in the foundation-model setting [3]. Our contribution is not batch-level IF in isolation. It is the integration of batch-level IF with a submodular diversity-quality objective (Eq. 9) and a small-proxy estimator under an explicit error bound (Theorems 6 and 7), together with a multi-GPU realization that the above methods do not target. We note that the key premise of our Theorem 6, that instances within a semantic cluster exhibit similar gradients and influences, is independently adopted by the concurrent clustering-by-gradient-similarity method [1], which supports the assumption empirically.

*Data support.* On the candidate pools the k-centered clustering accounts for only 1.8-4.3% of the per-dataset selection wall-clock (gradient computation dominates at 75-85%); e.g., on WebTable clustering is 23.7s of 570.9s total selection (4.1%). The 3-10% runtime claim is preserved after this accounting.

Additional selection-compute context (added to Table III / Section V-C): measured in FLOPs, the full MELD-DS selection stage costs 13.4 PFLOP (0.5B proxy, batch-level IF), versus 424.7 PFLOP for LESS (32x more, full-7B per-sample gradients) and 25.3 PFLOP for QuRating (1.9x more, 1.3B rater over the full pool). This is the quantitative form of the "3-10% of traditional IF" claim and directly answers the LESS/DataInf comparison.

*Manuscript changes.* Table III with clustering column and accompanying text (blue); Section V-C and Section VII-C positioning (blue).

*References (this response).*
[1] "ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs." 2025. arXiv:2506.10288.
[2] M. Nikdan, V. Cohen-Addad, D. Alistarh, V. Mirrokni. "Influence Distillation: Efficient Data Selection at Scale via Influence Distillation." 2025. arXiv:2505.19051.
[3] K. Axiotis, V. Cohen-Addad, M. Henzinger, et al. "Data-efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond." ICML 2024.

### R2-2.2. Theorem 6 assumption under heterogeneous and noisy data

*Comment.* Theorem 6 assumes the within-batch gradient standard deviation sigma is low due to semantic similarity, which may not hold for heterogeneous, long-tailed, noisy data. An empirical sensitivity analysis under poor batch cohesion is needed.

*Response.* We have added a sensitivity analysis that directly stresses the sigma assumption. We degrade batch cohesion in two ways and measure both the batch-IF approximation error against per-sample IF and the downstream F1: first, by replacing the k-centered clustering with random clustering, which destroys within-cluster similarity; second, by injecting cross-task and cross-domain samples into clusters at increasing rates to inflate sigma. We also extend the batch-size sweep already in Table V and report the measured within-batch gradient standard deviation alongside it.

The error grows with sigma in the manner predicted by the O(sigma / (lambda sqrt(|B|))) bound, but two factors keep the downstream degradation gradual and bounded: the damping factor lambda in the inverse-Hessian term, and the diversity and quality components of Eq. 9 that retain a high-value floor independent of the influence term. Random clustering is markedly worse than k-centered clustering at matched |B|, which confirms that semantic clustering is the mechanism that lowers sigma rather than an untested assumption.

We additionally note a refinement of the bound's reading: the within-batch sigma is itself a function of |B| and grows as |B| increases (a larger batch admits less semantically homogeneous members), so the error behaves as a sigma(|B|)/sqrt(|B|) tradeoff rather than decreasing monotonically in |B|. This explains the empirical sweet-spot already visible in the Table V batch-size sweep, and we have restated Theorem 6's discussion accordingly.

*Data support.* Across three datasets (RE, ER, CTA) the batch-IF approximation error rises monotonically with the measured within-batch gradient sigma; semantic k-centered clustering lowers both sigma and the resulting error by roughly 3.5x relative to random clustering at |B|=8; sigma(|B|) is confirmed to increase with |B|. The premise that within-cluster gradients are similar is also adopted independently by recent clustering-by-gradient-similarity selection [1].

*Manuscript changes.* Section V-C, after Theorem 6 (blue); Figure [X] (new); Table V extended with sigma (blue).

*References (this response).*
[1] "ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs." 2025. arXiv:2506.10288.

### R2-2.3. Baseline clarity: matched dense 7B, and the LLM Baseline column

*Comment.* (i) Mixtral differs from MELD in architecture and training purpose, so it does not cleanly isolate the standalone-router gain; a matched dense multi-task 7B baseline is preferable. (ii) The configuration of the LLM Baseline column in Table I is unclear.

*Response.* We address both parts.

For (i), we have added a matched dense multi-task 7B baseline to Table I. It is a single Mistral-7B, the same backbone used by the MELD experts, jointly fine-tuned over all ten tasks without the MoE structure and without the standalone router. This is the clean ablation that isolates the benefit of the MoE structure and routing from task specialization. We note that this baseline is distinct from the existing MELD w/o MoE entry in the ablation, which is a separate single expert fine-tuned per task; the new baseline is one model trained over all tasks jointly.

We report the finding directly and frame it honestly. In static in-distribution evaluation the matched dense multi-task 7B reaches parity with MELD (differences within noise across the task suite). This is the regime that Theorem 2 predicts parity for: with a full-capacity 7B backbone and adequate per-task data at a modest task count, the dense and MoE error bounds coincide; the bounds separate only in the low-capacity, low-per-task-data, or large-task-count regime. We therefore do not claim an in-distribution accuracy advantage for the MoE. The MoE design is justified instead by modular extensibility under accuracy parity: per-expert refinement and debugging, incremental onboarding of new tasks without retraining over the full data union and without cross-task regression, and plug-and-play top-m composition for held-out datasets. We have revised the Section VI-A discussion and the abstract accordingly, replacing the earlier wording that suggested an accuracy superiority of the MoE design with this parity-plus-extensibility framing, and we cross-reference the marginal-cost analysis (Table III) and the cross-dataset composition results (Table VI).

*Data support.* Matched dense multi-task 7B vs MELD over the shared tasks: mean per-task F1 is within +/-0.01 (e.g., dense 5-epoch is +0.01 on average; on the ER/DC subset used for the scaling study, dense at a 10x-reduced step budget reaches 0.860/0.868/0.865 at n=3/5/7 versus MELD 0.892, with no monotone interference decay). The remaining per-task gaps are confined to the most discriminative task (ER, +0.094 for MELD) and vanish on saturated tasks (DC, within +/-0.01), and even the ER gap is compute-confounded because MELD used roughly 10x the training compute; we therefore present this as parity rather than as an efficiency win.

For (ii), we have clarified the LLM Baseline column. It reports, per task, the strongest task-appropriate offline LLM baseline among the LLM-based methods we compare: JellyFish-13B for entity matching, data cleaning, error detection, and data imputation; TableLLaMa-7B for column type annotation, relation extraction, and entity linking; and ExtractGPT for attribute value extraction. Each is fine-tuned on its target task. We have added this specification to Section VI-A and a note to the Table I caption, and we make explicit that it is distinct from the newly added matched dense multi-task 7B model.

*Manuscript changes.* Table I with the dense 7B column (blue); Section VI-A and Table I caption (blue); abstract superiority-wording softened to parity-plus-extensibility (blue).

### R2-2.4. Standalone versus integrated MoE tradeoff

*Comment.* The tradeoff against jointly trained integrated MoE is not fully characterized. Integrated MoE may benefit from stronger shared representation learning in earlier layers; it is unclear whether the standalone design limits transfer across closely related tasks that differ mainly in output format or supervision signal. A more explicit analysis of when the standalone design helps and when it may hurt would strengthen the paper.

*Response.* We have added an explicit when-it-helps-versus-when-it-hurts analysis. The standalone design is most beneficial when tasks are heterogeneous and differ substantially in output domain, for example binary entity matching versus multi-class column type annotation versus generative data imputation. In that regime the decoupled experts specialize without interfering with one another, which the small cross-dataset and cross-task degradation in Table VI demonstrates. The standalone design is least advantageous for closely related tasks that differ mainly in output format or supervision signal, where an integrated MoE can exploit shared representation learning in earlier layers to be more sample-efficient; the gains from early-layer sharing, however, come with the joint-training instability discussed in our response to R1-2 [1], [2]. With the matched dense multi-task 7B baseline now in Table I and the MELD-with-Mixtral-router entry in Table VII, we show that on such closely related tasks MELD remains within a small margin of the integrated alternative while retaining plug-and-play composition and independent per-expert refinement. We have added this analysis to Section VI-D and connected it to the failure-case analysis in our response to R1-5.

*Data support.* Plug-and-play top-m composition on held-out ER datasets recovers approximately the best single related expert (mean top-m minus best-single approximately -0.01), clearly exceeds the mean single expert (+0.085 on average) and avoids the catastrophic worst single expert (+0.43 on abt-buy, +0.41 on walmart-amazon). The value is robustness to not knowing the right expert in advance, not an accuracy gain over the oracle best single.

*Manuscript changes.* Section VI-D (blue); cross-reference to Table I and Table VII (blue).

*References (this response).*
[1] Z. Chi, L. Dong, S. Huang, et al. "On the Representation Collapse of Sparse Mixture of Experts." NeurIPS 2022. arXiv:2204.09179.
[2] D. Dai, C. Deng, C. Zhao, et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." 2024. arXiv:2401.06066.

### R2-3.1. Figure 1 transformation: automated or manual

*Comment.* The Data Transform example from EM to DI is clever, but it should be clarified whether the transformation is automated or requires manual rule definition, since it affects the low-resource claim in terms of human effort.

*Response.* The EM-to-DI transformation in Figure 1 is fully automated and requires no per-instance manual rules. Given a labeled EM pair, a fixed attribute-masking operator removes one aligned attribute value and reuses the matched counterpart as the imputation label. The source-to-target mapping is specified once per task pair at the schema level, namely which attribute to mask and how to relabel, and is then applied automatically to all instances. No human effort scales with data size, so the low-resource claim is preserved. We have added this clarification to Section II-B and Section IV-A.

*Manuscript changes.* Section II-B and Section IV-A (blue).

### R2-3.2. Operationalization of the Eq. 1 information-bottleneck objective

*Comment.* Computing the mutual information in Eq. 1 is difficult in high-dimensional neural spaces. Please clarify whether it is directly estimated, approximated by a surrogate loss, or used mainly as conceptual guidance.

*Response.* We have clarified that Eq. 1 is used as the conceptual objective and is not estimated as an explicit mutual-information quantity. It is operationalized through the surrogate procedure in Section IV-C. The inner maximization over the expert parameters is realized by parameter-efficient fine-tuning that maximizes label likelihood on the augmented data. The outer minimization over the RAG parameters is enforced implicitly by controlling the amount of external data added per iteration. No high-dimensional mutual-information estimator is required; the information-bottleneck view provides the design rationale for balancing fit against generalization. We have added this clarification immediately after Eq. 1.

*Manuscript changes.* Section IV-C, after Eq. 1 (blue).

### R2-3.3. PPL-ratio thresholding and the value of contradictory samples

*Comment.* The thresholding f(x) for the PPL ratio assumes that anything with a ratio above one or near zero is useless, which is a heuristic. Is there a scenario where contradictory samples, as hard negatives, are beneficial for robustness?

*Response.* We agree that the thresholding is a conservative heuristic, and we have added a discussion. Contradictory or hard-negative samples can be beneficial when paired with a contrastive objective; near-OOD negatives provide more informative supervision than cross-task negatives [1]. In our setting, however, the PPL ratio is computed at the supervised-selection stage rather than within a contrastive objective, and a ratio above one most often reflects label noise introduced by self-annotation rather than an informative hard negative. Admitting such samples into supervised fine-tuning degrades calibration, so we exclude them at selection time. Exploiting hard negatives through an auxiliary contrastive term is a compatible extension; our noisy-augmentation robustness study (response to R1-3) partially probes this regime by characterizing how performance responds to increasing inconsistent content. We have added this discussion to Section V-B.

*Manuscript changes.* Section V-B, after Eq. 6 (blue).

*References (this response).*
[1] T. Yan, et al. "Contrastive Instruction Tuning." Findings of ACL 2024. arXiv:2402.11138.

### R2-3.4. Initialization-time difference in Table III

*Comment.* The Initialize Time for DataInf (1200s) versus MELD-DS (115s) needs more explanation. What specific part of the initialization was removed or optimized?

*Response.* We have added an explanation. The difference stems from what each method must precompute before any influence score is produced. DataInf computes and stores per-sample gradients of the full 7B target model over the entire candidate pool, which accounts for both its 1200-second initialization and its 22 GB gradient storage. MELD-DS warms up a 0.5B proxy model on the submodular subset S'_init and computes gradients only at proxy scale and only at the batch level, which reduces the gradient-storage footprint to 3.4 GB and the initialization time to 115 seconds. The reduction is therefore attributable to proxy-scale, batch-level gradient computation, and no step of the influence estimation itself is omitted. We have added this explanation to the Table III discussion in Section V-C.

*Manuscript changes.* Section V-C, Table III discussion (blue).

---

## Consolidated list of new references

> 落 bib 时确认格式；以下编号仅为本清单内部编号，response 各块用块内 [1][2][3]。

- A. Aghajanyan, S. Gonen, L. Zettlemoyer. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL-IJCNLP 2021. arXiv:2012.13255.
- Z. Chi, L. Dong, S. Huang, D. Dai, S. Ma, B. Patra, S. Singhal, P. Bajaj, X. Song, X.-L. Mao, H. Huang, F. Wei. "On the Representation Collapse of Sparse Mixture of Experts." Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 34600-34613. arXiv:2204.09179.
- D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen, et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." 2024. arXiv:2401.06066.
- T. Yan, et al. "Contrastive Instruction Tuning." Findings of the Association for Computational Linguistics: ACL 2024. arXiv:2402.11138.
- "ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs." 2025. arXiv:2506.10288.
- M. Nikdan, V. Cohen-Addad, D. Alistarh, V. Mirrokni. "Influence Distillation: Efficient Data Selection at Scale via Influence Distillation." 2025. arXiv:2505.19051.
- K. Axiotis, V. Cohen-Addad, M. Henzinger, S. Jerome, V. Mirrokni, D. Saulpic, D. P. Woodruff, M. Wunder. "Data-efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond." ICML 2024.

---

## `[TBD]` 状态（实测填充情况，详见 docs/REVISION_PER_QUESTION_ANNOTATION.md）

| 位置 | 待填内容 | 状态 |
|---|---|---|
| R2-2.1 | k-center 聚类 wall-clock + 占选择时间百分比 | 已填：1.8-4.3%（WebTable 23.7s/570.9s=4.1%）+ 选择 FLOP 13.4/424.7/25.3 PFLOP |
| R2-2.2 | error-vs-σ 曲线；random vs k-center 在 \|B\|=8 的差 | 已填：error 随 σ 单调升，k-center 比 random 低约 3.5×；σ(\|B\|) 递增 |
| R2-2.3 | dense multi-task 7B 各任务数值 | 已填并更正：parity（±0.01），非 MELD 超越；reframe 到 modularity |
| R1-3 | noisy 0/10/20/30% 下 MELD-DS vs full/DSIR/SuperFilter F1 降解 | 已填：ppl 过滤捕获 94.3% 注入噪声，降解约慢 2× |
| R1-4 | n∈{3,5,7,10} per-task F1、利用率、vs Mixtral | 已填：per-task F1 平稳（无干扰衰减），router 3→18 稳定 ≥0.95 |
