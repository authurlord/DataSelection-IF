# TKDE-2025-09-3230 审稿意见整理（中文翻译 + 论文位置 + Solution）

> 论文：*Efficient Mixture of Experts for Low Resource Fast Training Large Language Model Data Preprocessing*（MELD / MELD-DS），Major Revision，deadline 2026-08-31。
> 「论文位置」中带 ⚠ 的为根据上下文推断的大致位置，需对照 PDF 最终确认。
> Solution 中的 **E1–E9 / DEFER** 对应 `REVISION_PLAN.md` 中的实验编号。

---

## 一、Associate Editor 综合意见

### AE1 — ITS 假设在非表格领域的验证
- **原文**：The claim that heterogeneous DP tasks share a common intrinsic subspace is insufficiently validated beyond tabular tasks. The authors must test or discuss how this assumption holds for structurally different domains.
- **中文**：异构 DP 任务共享同一内在子空间（intrinsic task subspace, ITS）的声明，仅在表格类任务上验证不足。必须测试或讨论该假设在结构差异更大的领域是否成立。
- **论文位置**：⚠ 方法动机部分（meta-path / 共享子空间假设，Sec. III–IV）；ITS 概念处。
- **Solution**：**DEFER（他 repo）**。两条出路：①有时间则在非表格任务（text classification / NER，如 `datasets/` 下 events_classification_biotech、GLUE）上补 t-SNE / subspace-overlap 证据；②资源不足则降级为 Discussion/Limitation，用现有表格任务子空间可视化 + 文献讨论回应。

### AE2 — Theorem 6 在 batch cohesion 差时的敏感性
- **原文**：Theorem 6 assumes low within-batch gradient variance due to semantic similarity, which may not hold for noisy or heterogeneous real-world data. An empirical sensitivity analysis under poor batch cohesion is required.
- **中文**：Theorem 6 假设 batch 内梯度方差低（源于语义相似），但在噪声/异构真实数据上未必成立。需要在 batch cohesion 差时的经验敏感性分析。
- **论文位置**：Theorem 6（MELD-DS batch-level IF 近似的理论依据，Sec. V）。
- **Solution**：**E2（类型 P，仅 0.5B proxy）**。构造不同 cohesion：k-center vs random clustering + 向 cluster 注入噪声样本；测量 within-batch 梯度 σ、batch-IF vs per-sample IF 近似误差、下游选择质量曲线。选 1 个内聚数据集（RE/RE，梯度仅 ~133s）+ 1 个异质数据集，加 1 张 figure。

### AE3 — 聚类开销未计入 runtime
- **原文**：The runtime advantage of the batch-level IF approximation does not account for k-centered clustering overhead, which may be non-trivial on low-resource hardware. A full cost breakdown must be provided.
- **中文**：batch-level IF 的运行时优势未计入 k-centered clustering 开销，在低资源硬件上可能不可忽略。需提供完整的成本拆解。
- **论文位置**：Table III（效率对比：VRAM Peak / Initialize Time）。
- **Solution**：**E1（类型 C+W）**。`eval_result/time_dict_*.npy` **已把聚类阶段 `batch-division` 单独计时**（如 WebTable 聚类≈23.6s、梯度≈475s、IF≈49s）；聚类是纯 CPU（sklearn KMeans+Agglomerative）。在 Table III 增列分阶段 wall-clock + VRAM，明确聚类占比很小。

### AE4 — LLM Baseline 描述 + matched dense 7B
- **原文**：The "LLM Baseline" in Table I is insufficiently described. A matched dense multi-task 7B model should be included to isolate the benefit of the standalone router from other confounding factors.
- **中文**：Table I 的"LLM Baseline"描述不足。应加入一个 matched dense multi-task 7B 模型，以把 standalone router 的收益与其他混杂因素隔离开。
- **论文位置**：Table I（主结果表）及其 "LLM Baseline" 行；Sec. VI-A 实验设置。
- **Solution**：**E3（类型 T）+ E9（类型 W）**。E3：同一 Mistral-7B 不走 MoE，合并所有任务 train-select 联合微调 1 个 LoRA，逐任务 eval，加入 Table I。E9：补一段说明 LLM Baseline 的确切配置（Jellyfish-13B / TableLLaMa-7B / ExtractGPT 按任务，需作者确认）。

### AE5 — routing 不稳定性 / scalability / standalone vs integrated
- **原文**：Potential routing instability, scalability as task count grows, and scenarios where the standalone design may underperform integrated MoE architectures are not adequately discussed.
- **中文**：未充分讨论潜在的路由不稳定性、任务数增长时的可扩展性，以及 standalone 设计可能不如 integrated MoE 的场景。
- **论文位置**：Sec. IV（MELD 架构 / standalone router）；Table VII（消融，含 with-Mixtral）。
- **Solution**：**DEFER（他 repo）**。router 代码在另一 repo：补 routing entropy/confidence 稳定性指标、task 数 3→5→7→10 的 routing 质量与 expert utilization、以及 standalone 何时优/何时劣的讨论。

### AE6 — 自标注增强数据偏差 / 噪声鲁棒性
- **原文**：The reliance on self-annotated augmented data may introduce bias; stronger evidence of robustness under noisy augmentation is needed.
- **中文**：对自标注增强数据的依赖可能引入偏差；需要在噪声增强下更强的鲁棒性证据。
- **论文位置**：⚠ Sec. III/V 数据增强（self-annotation / $M_{RAG}$ 伪标注）部分。
- **Solution**：**E4（类型 T）**。向 train-select 的 output 注入 10/20/30% label noise，重训并画性能降解曲线 vs baselines；用分类型任务 ER/DC/SM。证明 MELD-DS 的选择对噪声增强具鲁棒性。

---

## 二、Reviewer 1（评分 Good）

### R1.1 — ITS 非表格领域验证（同 AE1）
- **原文**：The core assumption that heterogeneous data preprocessing tasks can be embedded into a shared intrinsic task subspace can be further validated. Except tabular tasks, whether this assumption holds for more complex or structurally different domains.
- **中文**：异构 DP 任务可嵌入共享内在子空间这一核心假设可进一步验证。除表格任务外，该假设在更复杂或结构不同的领域是否成立。
- **论文位置**：同 AE1。
- **Solution**：**DEFER（他 repo）**，同 AE1。

### R1.2 — standalone router 复杂度与不稳定性
- **原文**：For advantage of the standalone router over standard MoE routing mechanisms, additional training complexity and potential instability of routing are not carefully analyzed.
- **中文**：关于 standalone router 相对标准 MoE 路由机制的优势，其额外训练复杂度与潜在路由不稳定性未被仔细分析。
- **论文位置**：Sec. IV（standalone router）。
- **Solution**：**DEFER（他 repo）**，并入 AE5。

### R1.3 — 自标注/增强偏差与噪声鲁棒性（同 AE6）
- **原文**：Whether the reliance on self-annotated and augmented data introduces potential bias and noise? ... add sufficient evidence on robustness under noisy or biased augmentation.
- **中文**：自标注与增强数据是否引入潜在偏差和噪声？即使有所提的数据选择方法，也应补充在噪声/有偏增强下的鲁棒性证据。
- **论文位置**：同 AE6。
- **Solution**：**E4（类型 T）**，同 AE6。

### R1.4 — 任务数增长的可扩展性
- **原文**：As the number of tasks increases, managing expert specialization and maintaining routing quality may become challenging?
- **中文**：随着任务数增加，管理 expert 专门化、维持路由质量是否会变得困难？
- **论文位置**：Sec. IV（router / expert 管理）。
- **Solution**：**DEFER（他 repo）**，并入 AE5（task 数 scalability 曲线 + expert utilization）。

### R1.5 — failure case 分析
- **原文**：For the experimental evaluation, it would be good to enhance the analysis of failure cases.
- **中文**：实验评估中，建议加强 failure case 分析。
- **论文位置**：⚠ Sec. VI 实验分析部分。
- **Solution**：**E5（类型 W+C，无需 GPU）**。逐样本预测 CSV 已在 `output/Ablation/**/*.csv`；挑 main 最低分 case（DC/rayyan F1≈0.84、CTA/SimTab、CTA/WebTable）做错误归类。

---

## 三、Reviewer 2（评分 Fair）

### R2.1 — 相对 LESS/DataInf 的新颖性 + 聚类开销
- **原文**：The novelty must be carefully weighed against existing works like LESS and DataInf. ... the computational cost of the k-centered clustering ... is not fully accounted for in the runtime analysis. If $k$ is large, the clustering step itself may become a non-negligible overhead.
- **中文**：batch-level IF 的新颖性需相对 LESS、DataInf 审慎权衡。k-centered clustering 的计算成本未完全计入运行时分析；若 $k$ 大，聚类本身可能成为不可忽略的开销。
- **论文位置**：Sec. V（MELD-DS batch-IF）；Table III（runtime）。
- **Solution**：**E1（类型 C+W）+ E2 佐证**。E1 给出含聚类的完整 cost breakdown（聚类占比小）；E2 的敏感性结果同时强化「batch-IF 在何条件下成立」的 novelty 论证。写作上补一段相对 LESS/DataInf 的定位区分。

### R2.2 — Theorem 6 的 σ 假设（同 AE2）
- **原文**：Theorem 6 assumes that $\sigma$ (the standard deviation of gradients within a batch) is low due to semantic similarity ... many real-world DP datasets are heterogeneous, long-tailed, and noisy ... does not provide an empirical sensitivity analysis.
- **中文**：Theorem 6 假设 batch 内梯度标准差 σ 低（因语义相似），但真实 DP 数据常异构、长尾、有噪；语义相似不必然意味低梯度方差。缺少经验敏感性分析。
- **论文位置**：Theorem 6（Sec. V）。
- **Solution**：**E2（类型 P）**，同 AE2。

### R2.3 — baseline 清晰度（matched dense 7B + LLM Baseline 配置）
- **原文**：A stronger baseline would be a matched dense multi-task 7B model ... Table I includes an "LLM Baseline," but its exact configuration is not sufficiently clear ... specify whether this is a single dense 7B model jointly fine-tuned over all tasks, or some other setup.
- **中文**：更强的基线应是 matched dense multi-task 7B 模型（同设置下训练）以分离 MoE 结构收益与任务专门化收益。Table I 的"LLM Baseline"配置不够清楚，应说明是否是单一 dense 7B 在所有任务上联合微调。
- **论文位置**：Table I 及 Sec. VI-A 设置。
- **Solution**：**E3（类型 T）+ E9（类型 W）**，同 AE4。

### R2.4 — standalone vs integrated MoE 的 tradeoff
- **原文**：The tradeoff against jointly trained integrated MoE architectures is not fully characterized ... whether the standalone design may limit transfer across closely related tasks that differ mainly in output format or supervision signal. A more explicit analysis of when the standalone design helps and when it may hurt.
- **中文**：未充分刻画与 jointly-trained integrated MoE 的 tradeoff；尤其 standalone 设计是否会限制在「主要差异在输出格式/监督信号」的 closely related 任务间的迁移。需要更明确分析 standalone 何时有益、何时有害。
- **论文位置**：Sec. IV；Table VII（with-Mixtral 行）。
- **Solution**：**DEFER（他 repo）**，并入 AE5。可基于 Table VII 现有 Mixtral 对比数据扩展 closely related tasks（如 EM 与 DC）的讨论。

---

### R2 具体意见（Specific comments）

### R2.S1 — Fig.1 EM→DI 转换是否自动
- **原文**：Figure 1: The "Data Transform" example from EM to DI is clever. However, the authors should clarify if this transformation is automated or requires manual rule definition.
- **中文**：Fig.1 中 EM→DI 的"Data Transform"示例很巧妙，但应澄清该转换是自动的还是需要手工定义规则；若需任务专属手工规则，会影响"low-resource"在人力成本意义上的声明。
- **论文位置**：Figure 1（框架图 / Data Transform 示例）。
- **Solution**：**E8（类型 W）**。说明是基于规则的**自动**属性映射（DI 按 `ATTR_DICT` mask 指定属性，见 `README_dataset.md`），非逐例手工，人力成本可控。

### R2.S2 — Eq.1 IB 目标的操作化
- **原文**：Eq. 1: ... calculating the mutual information $I(M_G(X_i); M_G(\mathbb{X}_i))$ is notoriously difficult ... clarify how Eq. 1 is operationalized in practice, e.g., whether the MI objective is directly estimated, approximated by a surrogate loss, or used mainly as conceptual guidance.
- **中文**：Eq.1 基于 Information Bottleneck 的 Min-Max 目标理论上成立，但高维神经空间中的互信息难以计算。应说明实践中如何操作化：是直接估计 MI、用代理损失近似，还是主要作概念指导。
- **论文位置**：Eq. (1)（expert refinement 的 IB Min-Max 目标，Sec. IV）。
- **Solution**：**E7（类型 W）**。说明 MI 不直接估计，实际用 iterative augmentation + contrastive surrogate loss 近似，IB 主要作概念指导。

### R2.S3 — PPL ratio 阈值 f(x) 与 hard negative
- **原文**：Section V-B: ... quality score $F_{qual}$ based on PPL ratios. The thresholding $f(x)$ ... assumes that anything with a ratio $>1$ (contradictory) or near $0$ (too simple) is useless. This is a heuristic. Is there a scenario where "contradictory" samples (hard negatives) are actually beneficial for robustness?
- **中文**：Sec. V-B 基于 PPL ratio 的质量分 $F_{qual}$，其阈值函数 f(x) 假设 ratio>1（矛盾）或接近 0（过简）的样本无用，这是启发式的。是否存在"矛盾"样本（hard negatives）实际有利于鲁棒性的场景？
- **论文位置**：Section V-B（$F_{qual}$ / PPL ratio / f(x)）。
- **Solution**：**E6（类型 W，主；可选 T）**。承认 f(x) 为 heuristic，讨论 hard negative 可能有益的场景与当前保守设计的取舍；可选小实验：放宽阈值保留部分 ratio>1 样本对比（PPL 分数已存，选择为 CPU，只重训占 GPU）。

### R2.S4 — Table III DataInf 初始化 1200s vs 115s
- **原文**：Table III: The "VRAM Peak" for MELD-DS (1-GPU) is 14G ... However, the "Initialize Time" for DataInf (1200s) vs MELD-DS (115s) needs more explanation. What specific part of the initialization was removed/optimized?
- **中文**：Table III 中 MELD-DS 单卡 VRAM 峰值 14G 令人印象深刻、支持 RTX 3090 声明；但 DataInf(1200s) 与 MELD-DS(115s) 的初始化时间差异需更多解释——具体移除/优化了哪部分初始化？
- **论文位置**：Table III（Initialize Time 行）。
- **Solution**：**E1（类型 W）**。说明 MELD-DS 用 0.5B proxy model warm-up + 批级梯度，省去了 DataInf 在 full 7B 上逐样本梯度的存储与计算（结合 E1 分阶段计时数据佐证）。

---

## 四、汇总：编号 → Solution → 类型 → 论文位置

| 编号 | Solution | 类型 | 论文位置 |
|------|----------|------|----------|
| AE1 / R1.1 | DEFER（他 repo，ITS 非表格） | — | Sec. III–IV ⚠ |
| AE2 / R2.2 | **E2** Theorem 6 敏感性 | P | Theorem 6 |
| AE3 | **E1** cost breakdown | C+W | Table III |
| AE4 / R2.3 | **E3** dense 7B + **E9** 描述 | T+W | Table I |
| AE5 / R1.2 / R1.4 / R2.4 | DEFER（他 repo，router/MoE） | — | Sec. IV / Table VII |
| AE6 / R1.3 | **E4** noisy augmentation | T | Sec. III/V ⚠ |
| R1.5 | **E5** failure case | W+C | Sec. VI ⚠ |
| R2.1 | **E1** + **E2** + 写作定位 | C+W/P | Sec. V / Table III |
| R2.S1 | **E8** Fig.1 自动化说明 | W | Figure 1 |
| R2.S2 | **E7** Eq.1 IB 操作化 | W | Eq. (1) |
| R2.S3 | **E6** PPL f(x) 讨论 | W/(T) | Sec. V-B |
| R2.S4 | **E1** DataInf 对比解释 | W | Table III |

> 类型：W=纯写作 · C=只占CPU · T=从 train-select 重训(7B GPU) · P=重算梯度但仅 0.5B proxy
