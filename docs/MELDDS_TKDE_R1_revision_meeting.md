# MELD-DS (TKDE-2025-09-3230) Major Revision 评审意见整理 & Revision 讨论稿

> 状态：AE + 2 位审稿人，**R1 Good / Major、R2 Fair / Major**，两位都给 Major Revision（非 Reject），核心工作被认可。
> 窗口：决定 **02-Jun-2026** → 截止 **31-Aug-2026**，从今天（14-Jun）起约 **11 周**。
> 本文档：意见全文（英文原文 → 中文翻译 → 与论文内容的对应）→ 处理方案分类（A 写作 / B 现有数据重分析 / C 补 GPU 实验）→ 11 周排期 → 待讨论决策点 → 新增参考文献与文献支撑。

---

## 0. 总体判断（TL;DR）

- **没有出现"要求重做主实验"或"质疑方法正确性"级别的意见**。AE 的 6 条是 R1+R2 的合集，无第三方独立增量。
- 问题高度集中在三类：**补 1 个关键基线（dense 7B）+ 补 2~3 个鲁棒性/敏感性实验 + 一批写作澄清**。难度中等偏低，11 周足够（实测新实验在现有集群上约 2~3 周可跑完）。
- **真正需要 GPU 的新实验只有 4 个**：① matched dense multi-task 7B 基线（P0，AE 点名）② Theorem 6 batch-cohesion 敏感性（P0）③ noisy augmentation 鲁棒性（P1）④ task-count 可扩展性（P1）。ITS 非表格验证可降级为"小实验 + 文献论证"（P1/P2）。
- 其余（聚类开销拆分、failure case）用**已有 run 的日志/结果重分析**即可，零新 GPU。
- **关键杠杆**：
  - dense 7B 一个实验同时回应 **AE-4 + R2-2.3(a)**，并为 **R2-2.4**（standalone vs integrated 讨论）提供 dense 参考点；
  - Theorem 6 敏感性实验同时回应 **AE-2 + R2-2.2**；
  - noisy augmentation 同时回应 **AE-6 + R1-3**，并支撑 **R2-3.3**（contradictory 样本鲁棒性）。
- **已做文献调研，为 4 处说法找到支撑**（见 §7）：ITS 跨域成立、hard-negative 有益、router 不稳定性来自联合训练（反衬 MELD 解耦设计更稳）、batch-level/cluster IF 的定位与差异化。

---

## 1. Associate Editor 综合意见（6 条，均为 R1/R2 合集）

> AE 明确说明这 6 条是对两位 reviewer 意见的总结。这里给出原文+翻译+对应关系，**具体处理见对应的 R1/R2 条目**，避免重复。

| # | English (原文) | 中文翻译 | 映射到 | 论文位置 |
|---|---|---|---|---|
| AE-1 | The claim that heterogeneous DP tasks share a common intrinsic subspace is insufficiently validated beyond tabular tasks. The authors must test or discuss how this assumption holds for structurally different domains. | 异构 DP 任务共享同一内在子空间的声明，在表格任务以外验证不足。作者必须测试或讨论该假设在结构差异更大的领域如何成立。 | R1-1 | §III Thm 1, Q1；Fig 2；§II-B |
| AE-2 | Theorem 6 assumes low within-batch gradient variance due to semantic similarity, which may not hold for noisy or heterogeneous real-world data. An empirical sensitivity analysis under poor batch cohesion is required. | Theorem 6 假设因语义相似而 batch 内梯度方差低，这在噪声/异构真实数据上未必成立。需要在 batch cohesion 差时的经验敏感性分析。 | R2-2.2 | §V-C Thm 6 |
| AE-3 | The runtime advantage of the batch-level IF approximation does not account for k-centered clustering overhead, which may be non-trivial on low-resource hardware. A full cost breakdown must be provided. | batch-level IF 的运行时优势未计入 k-centered 聚类开销，在低资源硬件上可能不可忽略。需提供完整的成本拆分。 | R2-2.1 | §V-C；Table III |
| AE-4 | The "LLM Baseline" in Table I is insufficiently described. A matched dense multi-task 7B model should be included to isolate the benefit of the standalone router from confounding factors. | Table I 的 "LLM Baseline" 描述不足。应加入一个 matched dense multi-task 7B 模型，以从混淆因素中隔离 standalone router 的收益。 | R2-2.3 | §VI-A；Table I |
| AE-5 | Potential routing instability, scalability as task count grows, and scenarios where the standalone design may underperform integrated MoE are not adequately discussed. | 路由不稳定性、任务数增长时的可扩展性，以及 standalone 设计可能不如 integrated MoE 的场景，讨论不充分。 | R1-2, R1-4, R2-2.4 | §IV-C；§VI-G |
| AE-6 | The reliance on self-annotated augmented data may introduce bias; stronger evidence of robustness under noisy augmentation is needed. | 依赖自标注增强数据可能引入偏差；需要在噪声增强下更强的鲁棒性证据。 | R1-3 | §IV-A；§V |

---

## 2. Reviewer 1（Major Revision，评分 Good，预期修后接收）

> 全部为 1~2 个数量级以内的增量，**无质疑核心结论**。Reviewer 给出 "References sufficient / NA"，即不要求新增引用（我们补的文献是用来**加强论证**，非补缺口）。

### 意见与处理

#### R1-1：ITS 假设在非表格领域的验证【C 类（小实验）或 A 类（讨论+文献），见决策点 D1】

**English**：The core assumption that heterogeneous data preprocessing tasks can be embedded into a shared intrinsic task subspace can be further validated. Except tabular tasks, whether this assumption holds for more complex or structurally different domains.

**中文**：异构 DP 任务可嵌入共享内在任务子空间这一核心假设可进一步验证。除表格任务外，该假设在更复杂或结构不同的领域是否成立。

**与论文对应**：
- Theorem 1（§III）的依据 [36][40] **本身就是 NLP 跨任务工作**：Qin et al.（[36]，TASLP 2024）在 100 个 diverse NLP 任务上验证了统一 ITS；Hendel et al.（[40]，EMNLP 2023）的 task-vector 同样在 NLP 任务族上。即"ITS 不限于表格"在我们引用的文献里已被支撑，但**正文没把这一点讲透**，且 Fig 2 的 t-SNE 只画了 DP 任务。
- 缺的是：(a) 一句把"ITS 普适性来自 NLP 文献、我们把它实例化到 DP"说清楚的论证；(b) 可选的一个非表格/结构差异更大的任务点。

**处理方案**：
1. **文献论证（必做，A 类）**：在 §III Thm 1 讨论处加 2~3 句，明确 ITS 的普适性证据来自跨域 few-shot 文献，DP 是其在表格密集场景的实例化。引用 N1（Aghajanyan et al., ACL 2021，证明 task-specific 变化普遍局限于低维子空间，跨 GLUE 等多类 NLP 任务）+ 现有 [36]。**拟改草稿**：
   > "The existence of a low-dimensional intrinsic task subspace is not specific to tabular DP. Prior work shows that task-specific adaptation of pretrained models is confined to a low-dimensional subspace across a broad range of NLP tasks [36], [N1], and that soft prompts of up to one hundred heterogeneous tasks can be reparameterized within a single subspace of a few hundred dimensions [36]. Our use of the ITS assumption instantiates this property in the tabular-dense DP regime; Figure 2 visualizes the resulting per-task clustering."
2. **可选小实验（进取档，C 类，见 D1）**：补 1 个非表格 DP 任务（如 text-heavy 的 RE 已有，可再加一个更"文本推理"的任务，或在现有 t-SNE 上叠加一个结构差异大的任务族），展示其 task vector 仍落在同一 ITS 的不同簇。范围：1 任务 × t-SNE 复用现有 pipeline，约 1~2 天。
3. 若时间紧 → 进 Limitation：承认当前实证集中在表格/半结构化 DP，跨模态（如纯文本长推理）的 ITS 紧致性是正交未来工作。

**成本**：A 类 1h；进取档 +1~2 天。

#### R1-2：standalone router 的训练复杂度与不稳定性未分析【A 类：写作 + 文献支撑】

**English**：For advantage of the standalone router over standard MoE routing mechanisms, additional training complexity and potential instability of routing are not carefully analyzed.

**中文**：关于 standalone router 相对标准 MoE 路由机制的优势，其额外训练复杂度和潜在的路由不稳定性没有被仔细分析。

**与论文对应**：
- §IV-C "Router Network" 说明 router 与 MRAG 共享编码层、用 contrastive loss 训练；Theorem 3（§III）保证 router 能学到 ITS 簇。**但从未把"解耦式 standalone router 的训练复杂度/稳定性"与"联合训练的 built-in MoE"正面对比**。
- 这其实是我们的**强论点而非弱点**（见文献）：联合训练 MoE 的不稳定性主要来自 top-k 离散选择的不连续与 expert collapse（router 与 expert 在训练中相互拉扯）。MELD 把 expert 先训好再训固定专家上的 router，**结构上消除了这种 co-adaptation 不稳定**。

**处理方案**：在 §IV-C 末或 §VI-D（standalone vs built-in 讨论处）补一段**复杂度 + 稳定性**论证，引用 N4（Chi et al., NeurIPS 2022，形式化分析 sparse MoE 的 representation collapse，且随规模加剧）+ N5（DeepSeekMoE，靠 shared expert + bias 负载均衡缓解联合训练不稳定）。**拟改草稿**：
> "Compared with jointly trained MoE layers, the standalone router introduces no additional end-to-end training instability. Joint top-k routing is known to suffer from representation collapse and seed sensitivity caused by the discrete selection coupled with expert co-adaptation [N4], which production systems counteract with shared experts and bias-based load balancing [N5]. MELD decouples the two stages: experts are refined first, and the router is then trained over a fixed expert set with a contrastive objective. The router thus optimizes a stationary target, which removes the expert-router co-adaptation loop and the associated oscillation. Its only extra cost is one lightweight contrastive training pass that shares encoder layers with M_RAG; per-query inference cost is unchanged because only top-m experts are activated."

**成本**：1.5h（含文献）。

#### R1-3：自标注/增强数据的偏差与噪声鲁棒性【C 类：补实验（与 AE-6 合并）】

**English**：Whether the reliance on self-annotated and augmented data introduces potential bias and noise? Even with the proposed data selection method, it would be good to add sufficient evidence on robustness under noisy or biased augmentation.

**中文**：依赖自标注和增强数据是否引入潜在偏差和噪声？即使有所提出的数据选择方法，最好补充在噪声或偏差增强下鲁棒性的充分证据。

**与论文对应**：
- §IV-A 自标注 + §IV-B meta-path 增强是偏差来源；§V MELD-DS 的 quality/complexity 度量是设计上的"抗噪"答案，但**没有受控的 noisy-augmentation 鲁棒性曲线**。
- 现有 Table IV（ablation）间接说明 quality 分量的作用，但不是噪声鲁棒性实验。

**处理方案（实验，见 §5 排期 C-③）**：向 augmented pool 注入受控比例的标签噪声（如 0% / 10% / 20% / 30% 翻转或随机化 self-annotation 标签），对比 **MELD-DS vs 不做选择（full aug）vs 纯 diversity（DSIR）vs 纯 quality（SuperFilter）** 的下游性能降解曲线。
- 数据集：选 2~3 个代表性任务（如 ER 的 Amazon-Google + CTA 的 WebTable + DC 的 Rayyan）。
- 预期结论：MELD-DS 的 quality（PPL 比）+ complexity（IF）联合度量能过滤掉被注入的噪声，降解显著慢于 baseline → 直接坐实"对 noisy augmentation 鲁棒"。
- 产出：1 张降解曲线图（放主文或 appendix）+ response 文字。

**成本**：约 2~3 天（多比例 × 多任务，但单次 run 轻量）。

#### R1-4：任务数增长时的 expert specialization 与路由质量【C 类：补实验】

**English**：As the number of tasks increases, managing expert specialization and maintaining routing quality may become challenging?

**中文**：随着任务数增加，管理专家专精化和维持路由质量可能变得有挑战？

**与论文对应**：
- §VI-G 的 Fig 8 只 sweep 了 **expert 数 m**，没 sweep **任务数 n**。Theorem 2 的稀疏因子 s 给了"更稀疏更易泛化"的理论，但没有任务数维度的经验证据。
- 这是真实的缺口（reviewer 问的是 n 不是 m）。

**处理方案（实验，见 §5 排期 C-④）**：固定 expert 池，按任务数 n ∈ {3, 5, 7, 10} 逐步纳入任务，报告：(a) 各任务 F1 是否随 n 增大而退化（router 干扰）；(b) 路由命中率 / 专家利用率（load balance）随 n 的变化；(c) 与 built-in Mixtral 在相同 n 下的对比。
- 实验复用已训练专家，主要重训 router + 评估，成本可控。
- 预期：standalone router 因专家已固定专精，n 增大时 per-task 性能稳定、负载不塌缩；可对照 Theorem 3 的簇可分性。

**成本**：约 1~2 天。

#### R1-5：加强 failure case 分析【B 类：现有结果重分析】

**English**：For the experimental evaluation, it would be good to enhance the analysis of failure cases.

**中文**：对于实验评估，最好加强失败案例的分析。

**与论文对应**：
- §VI-B 提到 Mixtral 在 AVE/CTA 上超过 MELD，但**没有 case-level 失败归因**。Table I/II/VI 已有足量逐数据集结果可挖。

**处理方案（零新 GPU）**：从现有结果挑 2~3 个 MELD 表现不佳/被 baseline 反超的 case（如 AVE 的 OA-mine、CTA 的 SemTab19 Macro-F1、DI 的 Amazon），做归因分析：长上下文/开放域任务上 Mixtral 的 8 专家覆盖更广；MELD 在闭域任务强、开放域信息检索类任务相对弱（与 §VI-B 的现有论述一致，补成一小节 + 1 个表）。可同时呼应 R2-2.4。
- 产出：appendix 一小节"Failure Case Analysis" + 主文 1~2 句指针。

**成本**：约 0.5 天（CPU/分析）。

---

## 3. Reviewer 2（Major Revision，评分 Fair）

> R2 最具体、最技术。技术问题 2.1~2.4 + 具体问题 3.1~3.4。注意 R2 也给 "References sufficient / NA"。

### 技术问题

#### R2-2.1：batch-IF 的新颖性（vs LESS/DataInf）+ 聚类开销未计入【B 类（开销）+ A 类（定位）+ 文献】

**English**：The primary addition is the MELD-DS framework. While the batch-level IF approximation is a solid technical contribution, its novelty must be carefully weighed against existing works like LESS and DataInf. The authors claim a 3-10% runtime of traditional IF methods. However, the computational cost of the k-centered clustering required to form batches is not fully accounted for in the runtime analysis. If k is large, the clustering step itself may become a non-negligible overhead on low-resource hardware, and this cost should be explicitly accounted for.

**中文**：主要新增是 MELD-DS。batch-level IF 近似是扎实的技术贡献，但其新颖性须相对 LESS/DataInf 等审慎权衡。作者声称仅 3-10% 的传统 IF 运行时，但形成 batch 所需的 k-centered 聚类开销未充分计入；若 k 大，聚类本身在低资源硬件上可能成为不可忽略的开销，应显式核算。

**与论文对应**：
- §V-C 的"9× over DataInf、90× over LESS"、abstract 的"3-10% runtime"；Table III 的 "Selection Time" 列**没有单独列出聚类时间**。
- 新颖性方面：batch-level IF 本身并非全新（见文献，已有把 LESS 扩展到 batch-level 的工作；ClusterUCB 用"梯度相似→影响相似"做聚类）。我们的差异化贡献是**组合**：submodular S'_init 作 k-center 初始中心 + 0.5B proxy + 多 GPU 并行 + 把 IF 统一进 diversity/quality/complexity 的 submodular 目标（Eq.9）+ Theorem 6/7 的误差与渐近最优性保证。

**处理方案**：
1. **聚类开销核算（B 类，零新 GPU）**：在 Table III 增列或在文字中给出 k-center 聚类的 wall-clock。关键论据：贪心 k-center 是 O(nk)，且我们**用 S'_init 作初始中心**（聚类与选择共享计算），k 由选择预算决定（c=30%, |B|=8，故簇数 = |S'|/|B|，量级远小于 n）。在 14k 数据上聚类是秒级，相对 selection 的数百秒可忽略。**拟改草稿**：
   > "We additionally account for the clustering overhead. The k-centered clustering reuses the submodular subset S'_init as initial centers, so it shares computation with the selection stage rather than adding an independent pass. Greedy k-center runs in O(n k) time, and the number of clusters equals |S'|/|B|, which is far smaller than n under our budget (c = 30%, |B| = 8). On the 14,856-sample WebTable pool, clustering completes in [X] s, under [Y]% of the total selection time reported in Table III, and is therefore not a bottleneck on a single RTX 3090."
   （[X][Y] 待从现有 run 的 timer 补；若无 timer，单独计时一次即可，秒级。）
2. **新颖性定位（A 类 + 文献）**：在 §VII-C（LLM-Based Data Selection）或 §V-C 引言处补 1~2 句，承认 batch-level/cluster-level IF 的相关工作（N6 ClusterUCB、N7 Influence Distillation、N8 Axiotis et al. ICML 2024 的 clustering-based sensitivity sampling），并明确我们的差异点。**拟改草稿**：
   > "Cluster- and batch-level influence estimation has been explored concurrently: ClusterUCB [N6] groups samples by gradient similarity and allocates a scoring budget across clusters with a bandit policy, Influence Distillation [N7] computes exact influence on landmark samples and propagates it, and clustering-based sensitivity sampling provides provable guarantees in the foundation-model setting [N8]. Our contribution is not batch-level IF in isolation, but its integration with a submodular diversity-quality objective and a proxy-model estimator under an explicit error bound (Theorems 6 and 7), together with a multi-GPU realization that the above methods do not target."

**成本**：开销核算 ~2h；定位 ~1h。

#### R2-2.2：Theorem 6 的 σ 假设在异构/噪声数据上脆弱【C 类：补实验（与 AE-2 合并，P0）】

**English**：Theorem 6 ... assumes that σ (the standard deviation of gradients within a batch) is low due to semantic similarity. ... many real-world DP datasets are heterogeneous, long-tailed, and noisy. ... The current paper does not provide an empirical sensitivity analysis showing how MELD-DS behaves when batch cohesion is poor, when clusters are noisy, or when within-batch gradients are highly diverse.

**中文**：Theorem 6 假设 batch 内梯度标准差 σ 因语义相似而低；但许多真实 DP 数据异构、长尾、含噪，此时语义相似不必然意味低梯度方差。论文没有提供 batch cohesion 差、簇含噪、batch 内梯度高度发散时 MELD-DS 行为的经验敏感性分析。

**与论文对应**：
- §V-C Theorem 6：误差界 O(σ/(λ√|B|))，σ 是 batch 内梯度平均标准差。论文未给"σ 变大时性能/误差怎么变"的曲线。
- 这与 Theorem 6 的 σ 直接挂钩，是最该补的理论支撑实验之一。

**处理方案（实验，见 §5 排期 C-②，P0）**：构造受控的 batch cohesion 退化，测量 batch-IF 近似误差与下游性能：
- **退化手段**：(a) 把 k-center 聚类替换为 random clustering（破坏簇内相似性）；(b) 向簇内按比例注入跨任务/跨域样本（人为抬高 σ）；(c) 直接 sweep |B| ∈ {1,4,8,16,32}（已有 Table V 部分数据，可扩展并加入 σ 测量）。
- **测量**：每种设置下，估计 batch 内实际 σ（或梯度方差），对比 batch-IF 估计值 vs 逐样本 IF 的偏差，以及最终 F1。
- 预期：σ 升高时误差按 Theorem 6 的 O(σ/(λ√|B|)) 趋势上升，但因 K_min 风格的下限（这里是 quality/diversity 分量兜底）+ damping λ，性能退化平缓且可控；random clustering 明显差于 k-center → **正向验证"语义聚类降低 σ"是机制而非假设**。
- 产出：1 张"σ（或 cohesion 设置）vs 近似误差 / F1"图 + Theorem 6 讨论处 2~3 句。

**成本**：约 2~3 天（多设置，单 run 轻量，可与 Table V 复用）。

#### R2-2.3：baseline 公平性 + "LLM Baseline" 配置不清【C 类（dense 7B，P0）+ A 类（描述，待确认）】

**English**：(i) Mixtral differs not only in routing design but also in model architecture and training purpose ... A stronger baseline would be a matched dense multi-task 7B model trained under the same task setting. (ii) Table I includes an "LLM Baseline," but its exact configuration is not sufficiently clear ... specify whether this is a single dense 7B model jointly fine-tuned over all tasks, or some other setup.

**中文**：(i) Mixtral 不仅路由设计不同，架构与训练目的也不同，无法干净隔离 standalone router 的收益；应加入相同任务设定下训练的 matched dense multi-task 7B 基线。(ii) Table I 的 "LLM Baseline" 配置不清，应说明它是单个在所有任务上联合微调的 dense 7B，还是其他设置。

**与论文对应**：
- §VI-A 描述的 LLM-based 方法是 **JellyFish(13B)/TableLLaMa(7B)/ExtractGPT**（按任务族选用）。Table I 的单列 "LLM Baseline" 到底是"按任务取最佳 LLM 基线"还是"单个联合微调模型"，**正文确实没说死**（→ 待确认项 Q-A）。
- ⚠️ **重要区分**：现有 Table VII 的 "MELD w/o MoE" 是**每任务一个单专家**，**不等于** reviewer 要的"一个 dense 7B 在全部任务上联合微调"。后者是全新实验。

**处理方案**：
1. **matched dense multi-task 7B（实验，C 类，P0，见 §5 排期 C-①）**：用与 MELD 专家相同的 **Mistral-7B**，**不走 MoE、不用 standalone router**，在全部 10 个任务上**联合 SFT**一个模型，加入 Table I（新增一列）。这是 AE 点名要求、隔离"MoE 结构 + 路由专精"收益的干净 ablation。预期 MELD（多稀疏专家）> dense 多任务 7B（单密集模型），坐实 Theorem 2 的"多稀疏胜一密集"。
2. **"LLM Baseline" 描述（A 类，待 Q-A 确认后定稿）**：在 §VI-A 明确该列的确切构成（按任务族选用的最强 per-task LLM 基线，或其他）。**拟改草稿（占位，依 Q-A 调整）**：
   > "The 'LLM Baseline' column reports the strongest task-appropriate offline LLM baseline per task family (JellyFish-13B for EM/DC/ED/DI, TableLLaMa-7B for CTA/RE/EL, ExtractGPT for AVE), each fine-tuned on its target task. It is distinct from the newly added matched dense multi-task 7B model (Mistral-7B jointly fine-tuned over all tasks without MoE), which isolates the contribution of the MoE structure and the standalone router."

**成本**：dense 7B ~2~3 天（含联合训练 + 全任务评估）；描述 ~0.5h。

#### R2-2.4：standalone vs integrated MoE 的 tradeoff 未刻画【A 类：写作 + 现有数据 + 文献】

**English**：The tradeoff against jointly trained integrated MoE architectures is not fully characterized. ... integrated MoE models can potentially benefit from stronger shared representation learning in earlier layers ... they do not fully answer whether the standalone design may limit transfer across closely related tasks that differ mainly in output format or supervision signal. A more explicit analysis of when the standalone design helps and when it may hurt would strengthen the paper.

**中文**：与联合训练的 integrated MoE 的权衡未充分刻画。integrated MoE 可能受益于更早层的共享表示学习；现有跨数据集/跨任务实验未完全回答 standalone 设计是否会限制"主要在输出格式或监督信号上不同的近邻任务"之间的迁移。需要更明确地分析 standalone 何时有益、何时可能有害。

**与论文对应**：
- §VI-D 与 Table VII 的 "MELD with Mixtral"（用 Mixtral 替换 router）已有**部分对照数据**，§VI-B 也讨论了 Mixtral 在开放域/闭域上的强弱差异。但**没有"何时 standalone 帮、何时伤"的明确归因**，尤其是近邻任务（如 EM 与 DC，输出域都是二分类）。
- dense 7B（C-①）一旦补上，正好给这个讨论提供 integrated-style 的共享表示参考点。

**处理方案（A 类，零新 GPU）**：在 §VI-D 或 §VI-E 末补一段**"When standalone helps vs. hurts"**：
- **有益场景**：异构、输出域差异大的任务（EM 二分类 vs CTA 多分类 vs DI 生成）→ 解耦专家各自专精、互不干扰；MELD 在 C-D/C-T（Table VI）退化小即证。
- **可能受限场景**：仅在输出格式/监督信号上不同的近邻任务，integrated MoE 早层共享表示可能更省样本；用 Table VII（EM 各数据集）+ dense 7B 数据对照说明 MELD 此时与 integrated 差距很小、且换来部署灵活性（plug-and-play、单专家可独立 refine）。
- 文献：引用 N4/N5 说明 integrated MoE 的"早层共享表示"优势伴随联合训练不稳定的代价，形成 tradeoff 叙事。
- 同时呼应 R1-5 的 failure case。

**成本**：约 1.5h（含与 dense 7B 数据交叉）。

### 具体问题

#### R2-3.1：Figure 1 的 EM→DI 转换是自动还是手工规则？【A 类：纯写作】

**English**：The "Data Transform" example from EM to DI is clever. However, the authors should clarify if this transformation is automated or requires manual rule definition. If task-specific manual rules are required ... it affects the practical meaning of the "low-resource" claim in terms of human effort.

**中文**：EM→DI 的 "Data Transform" 例子很巧妙，但应澄清该转换是自动的还是需要手工规则定义。若需任务特定的手工规则，会影响"低资源"声明在人力成本上的实际意义。

**与论文对应**：§II-B（transform q→q′）+ Fig 1 + §IV-A（self-annotation/masking）。机制上是**固定的属性掩码 + 重标注**，但正文没把"自动、无逐实例人工"讲死。

**处理方案（拟改草稿）**：
> "The EM-to-DI transformation in Figure 1 is fully automated and requires no per-instance manual rules. Given a labeled EM pair, a fixed attribute-masking operator removes one aligned attribute value and reuses the matched counterpart as the imputation label. The source-to-target mapping is specified once per task pair at the schema level (which attribute to mask and how to relabel) and is then applied automatically to all instances, so no human effort scales with data size. This preserves the low-resource claim."

**成本**：0.5h。

#### R2-3.2：Eq.1 的 IB 互信息在实践中如何操作化？【A 类：纯写作】

**English**：Eq. 1: ... calculating the mutual information I(M_G(X_i); M_G(𝕏_i)) is notoriously difficult in high-dimensional neural spaces. The authors should clarify how Eq. 1 is operationalized in practice, e.g., whether the mutual-information objective is directly estimated, approximated by a surrogate loss, or used mainly as conceptual guidance.

**中文**：Eq.1 的互信息在高维神经空间中难以计算。应澄清 Eq.1 实践中如何操作化：是直接估计、用代理损失近似、还是主要作概念指导。

**与论文对应**：§IV-C Eq.1 后已有"In practice, we adopt an iterative optimization strategy..."，即**已经是代理实现**，但没明说"不直接估 MI"。

**处理方案（拟改草稿）**：
> "We clarify that Eq. 1 is used as the conceptual objective and is not estimated as an explicit mutual-information quantity. It is operationalized through the surrogate procedure in Section IV-C: the inner maximization over θ_{M_G} is realized by parameter-efficient fine-tuning that maximizes label likelihood on the augmented data, and the outer minimization over θ_{M_RAG} is enforced implicitly by controlling the amount of external data ΔX_i added per iteration. No high-dimensional mutual-information estimator is required; the information-bottleneck view provides the design rationale for balancing fit and generalization."

**成本**：0.5h。

#### R2-3.3：PPL 比阈值 f(x) 是启发式；contradictory（hard negative）样本是否有益？【B 类：写作 + 文献（可选小 ablation）】

**English**：Section V-B: ... The thresholding function f(x) for PPL ratios (0 ≤ x ≤ 1) assumes that anything with a ratio >1 (contradictory) or near 0 (too simple) is useless. This is a heuristic. Is there a scenario where "contradictory" samples (hard negatives) are actually beneficial for robustness?

**中文**：§V-B 的质量分 F_qual 基于 PPL 比；f(x) 假设 ratio>1（矛盾）或近 0（过简）都无用，这是启发式。是否存在"矛盾"样本（hard negative）实际上有益于鲁棒性的场景？

**与论文对应**：§V-B Eq.6 的 f(x)（0≤x≤1 取 x，否则 0）。当前设计**主动剔除 ratio>1**，确实偏保守。

**处理方案（写作 + 文献，可选小 ablation）**：承认是保守启发式，并基于文献给出有理由的讨论：hard negative 在对比式 instruction tuning 中确有价值（N3 COIN：来自 near-OOD 的 hard negative 提供比 far-OOD 更有信息量的监督信号），但其收益依赖**配对的对比目标**；MELD-DS 的 f(x) 工作在 SFT 数据选择阶段，剔除 ratio>1 是为了避免**自标注引入的真噪声/矛盾标签**污染 SFT，而非否认 hard negative 的价值。**拟改草稿**：
> "The thresholding f(x) is a conservative heuristic tailored to the SFT selection stage. Contradictory or hard-negative samples can be beneficial when paired with a contrastive objective: near-OOD negatives provide more informative supervision than cross-task negatives [N3]. In our setting, however, a PPL ratio above one most often reflects label noise introduced by self-annotation rather than an informative hard negative, and admitting such samples into supervised fine-tuning degrades calibration. We therefore exclude them at the selection stage. Exploiting hard negatives through an auxiliary contrastive term is a compatible and promising extension, which our noisy-augmentation robustness study (Section [VI-X]) partially probes."
- 可选小 ablation（若 D2 决定做）：保留一定比例 ratio∈(1, 1+ε] 的样本，看是否对鲁棒性有微增益；与 C-③ noisy 实验共用 pipeline。

**成本**：写作 + 文献 ~1h；可选 ablation +0.5 天。

#### R2-3.4：Table III 中 DataInf 初始化 1200s vs MELD-DS 115s，具体优化了什么？【A 类：纯写作】

**English**：Table III: ... the "Initialize Time" for DataInf (1200s) vs MELD-DS (115s) needs more explanation. What specific part of the initialization was removed/optimized?

**中文**：Table III 中 DataInf 初始化 1200s 与 MELD-DS 115s 的差异需更多解释，具体移除/优化了哪部分初始化？

**与论文对应**：§V-C + Table III。差异来自**proxy 规模 + batch 级梯度**，正文未点明。

**处理方案（拟改草稿）**：
> "The initialization gap stems from what each method must precompute before any influence score is produced. DataInf computes and stores per-sample gradients of the full 7B target model over the entire candidate pool, which accounts for both its 1200 s initialization and its 22 GB gradient storage. MELD-DS warms up a 0.5B proxy model on the submodular subset S'_init and computes gradients only at proxy scale and only at the batch level, reducing the gradient-storage footprint to 3.4 GB and the initialization time to 115 s. The reduction is therefore attributable to proxy-scale, batch-level gradient computation, not to omitting any step of the influence estimation itself."

**成本**：0.5h。

---

## 4. 处理方案分类汇总

### A 类：纯写作（零 GPU，合计 ~1 天）

| 意见 | 改动位置 | 文献 | 工作量 |
|---|---|---|---|
| R1-1（论证部分） | §III Thm 1 讨论 | N1 + 现有[36] | 1h |
| R1-2 | §IV-C / §VI-D router 复杂度+稳定性 | N4, N5 | 1.5h |
| R2-2.1（定位部分） | §VII-C / §V-C 新颖性差异化 | N6, N7, N8 | 1h |
| R2-2.3（描述部分） | §VI-A "LLM Baseline" 明确（待 Q-A） | — | 0.5h |
| R2-2.4 | §VI-D/E "when standalone helps vs hurts" | N4, N5 | 1.5h |
| R2-3.1 | §II-B/§IV-A Figure 1 自动化 | — | 0.5h |
| R2-3.2 | §IV-C Eq.1 操作化 | — | 0.5h |
| R2-3.3（讨论部分） | §V-B f(x) hard-negative 讨论 | N3 | 1h |
| R2-3.4 | §V-C/Table III 初始化时间解释 | — | 0.5h |

### B 类：现有数据/日志重分析（零新 GPU，合计 ~1 天）

| 意见 | 产出 | 工作量 |
|---|---|---|
| R2-2.1（开销核算） | Table III 增列/文字给 k-center 聚类 wall-clock（O(nk)，秒级） | 2h（含必要时单次计时） |
| R1-5 | appendix "Failure Case Analysis" 小节 + 主文指针 | 0.5d |

### C 类：补 GPU 实验（合计约 2~3 周窗口，单 run 均轻量）

| # | 实验 | 回应意见 | 预计成本 | 优先级 |
|---|---|---|---|---|
| ① | **matched dense multi-task 7B**（Mistral-7B 全任务联合 SFT，无 MoE/router）→ Table I 新列 | **AE-4 + R2-2.3(a)** + 喂 R2-2.4 | 2~3d | **P0** |
| ② | **Theorem 6 batch-cohesion 敏感性**（random vs k-center 聚类、注噪抬 σ、扩展 |B| sweep + σ 测量） | **AE-2 + R2-2.2** | 2~3d | **P0** |
| ③ | **noisy augmentation 鲁棒性**（注入 0/10/20/30% 标签噪声，MELD-DS vs full/DSIR/SuperFilter 降解曲线） | **AE-6 + R1-3** + 喂 R2-3.3 | 2~3d | **P1** |
| ④ | **task-count 可扩展性**（n∈{3,5,7,10}：per-task F1 / 负载 / 路由命中 / vs Mixtral） | **R1-4 + AE-5** | 1~2d | **P1** |
| ⑤ | （可选）**ITS 非表格任务点**（1 个文本推理类任务 + t-SNE 叠加） | R1-1, AE-1 | 1~2d | P1/P2（D1） |

> 杠杆：①同时服务 AE-4/R2-2.3/R2-2.4；②=AE-2+R2-2.2；③=AE-6+R1-3 且喂 R2-3.3。

---

## 5. 11 周排期（6/14 → 8/31）

| 周 | 任务 | 负责人(待分) |
|---|---|---|
| **W1（6/14–6/22）** | 确认 Q-A~Q-E（见 §6）；启动 C-① dense 7B 联合训练；清 A 类一半（3.1/3.2/3.4/2.3 描述）；response letter 骨架（逐条占位） | |
| **W2（6/23–6/29）** | C-① 出数入 Table I；C-② Theorem 6 敏感性跑起来；B 类聚类开销计时 + 写入 Table III | |
| **W3（6/30–7/6）** | C-② 出图 + Thm 6 讨论定稿；C-③ noisy augmentation 跑起来；A 类剩余（1-2 router、2-4 standalone、2-1 定位、3-3 讨论）配文献 | |
| **W4（7/7–7/13）** | C-③ 降解曲线出图；C-④ task-count 实验跑 + 出数；R1-5 failure case 分析 | |
| **W5（7/14–7/20）** | C-④ 收尾；C-⑤（若 D1 决定做）ITS 非表格点 + t-SNE；新结果写成正文（dense 7B 段、敏感性段、noisy 段、scalability 段） | |
| **W6（7/21–7/27）** | 文献 N1–N8 落地引用 + bib 核对（DOI/venue）；图表统一（scatter/曲线，"Better" 标注，multiplier 标注）；页数审计（§VII 压缩方案，见 D5） | |
| **W7（7/28–8/3）** | Response letter 全文初稿（逐条：改动摘要 + 节号/页码 + 新结果数据）；按 rebuttal 约定（顺序 [1][2][3]、无 em-dash、无 bold 子标题、表给全数据） | |
| **W8（8/4–8/10）** | manuscript 修改处统一**蓝色高亮**（TKDE 要求异色标注）；全文重编译，label/ref/编号/数据一致性审计 | |
| **W9（8/11–8/17）** | 合作者交叉审（Yaoshu/Jianxin）；response letter 与 manuscript 对齐核对 | |
| **W10（8/18–8/24）** | Buffer + 可选项（D2 hard-neg ablation、C-⑤ 若延期、D6 baseline 调优点） | |
| **W11（8/25–8/31）** | 最终 PDF↔source 一致性核对；TKDE 页数/格式合规（黑色文字、bio、index terms 齐全）；**提交** | |

**关键路径**：W1–W4 的 C-①②③④。dense 7B（①）是 P0 且 AE 点名，**W1 必须先启动**。

---

## 6. 待讨论决策点

| # | 问题 | 选项 | 我的倾向 |
|---|---|---|---|
| **D1** | R1-1（ITS 非表格）做到什么程度 | (a) 仅文献论证 + Limitation；(b) 文献 + 补 1 个非表格任务点 + t-SNE 叠加（C-⑤） | **(b)**，一个小实验把 reviewer "test or discuss" 的 "test" 也满足，更稳；若 W5 时间紧则退 (a) |
| **D2** | R2-3.3 是否做 hard-negative 小 ablation | (a) 仅写作+文献；(b) 加 ratio∈(1,1+ε] 保留的小 ablation（与 C-③ 共用） | **(a) 优先**，文献 N3 + 我们"剔除的是真噪声非信息性 hard-neg"的论证通常够；(b) 留 W10 buffer 可选 |
| **D3** | dense 7B（C-①）的 backbone | Mistral-7B（与 MELD 专家同源，最干净）vs 其他 | **Mistral-7B**，同源才"matched"，干净隔离 MoE/router 收益 |
| **D4** | C-② 退化手段取几种 | random-cluster / 注噪抬 σ / |B| sweep 三选几 | **三种都做但精简**：random vs k-center（机制证明）+ |B| sweep（扩 Table V，顺带 σ 测量）为主，注噪作补充 |
| **D5** | 页数：补内容放哪 | (a) 压 §VII Related Work（约 1 页→0.6 页）；(b) 新表/列合并进现有表；(c) 溢出进 full version [30] | **(a)+(b)+(c) 组合**。⚠️ **此为 memory「page-cutting」原则驱动的建议**：按你的偏好做**最小定向编辑**，§VII 压缩前我会先标出拟删句给你确认，不擅自删实质内容 |
| **D6** | 是否给最强 baseline 做 best-effort 调优点 | 做 / 不做（配置默认 + response 说明） | **先不做**，TKDE 这两位没质疑 baseline 配公平性到要求调优的程度；留 W10 可选 |
| **D7** | Q-A（"LLM Baseline" 到底是什么）确认前 | 用占位草稿 / 等你确认再写 | **等 Q-A 确认**再定稿该段（影响 R2-2.3 描述 + Table I 脚注） |

---

## 7. 新增参考文献与文献支撑（已做 survey）

> 说明：两位 reviewer 均答 "References sufficient / NA"，**不要求新增引用**。以下 N1–N8 用于**加强 rebuttal 论证**，按需引入。response letter 内按你的约定用**每个回应块独立的 [1][2][3] 顺序编号**（不与正文 [29] 等混用）；下表给出全量信息便于落 bib。**带 ⚠️ 的需在落 bib 前核对 arXiv 编号/DOI**。

| 标签 | 支撑哪条意见 | 文献 | 用途 / justify 的说法 |
|---|---|---|---|
| **N1** | R1-1, AE-1 | Aghajanyan, Gonen, Zettlemoyer. "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL-IJCNLP 2021. arXiv:2012.13255 | 任务特定适配普遍局限于**低维子空间**，跨多类 NLP 任务成立 → ITS 不限于表格 |
| **N2**（=现有[36]，强化用） | R1-1, AE-1 | Qin et al. "Exploring Universal Intrinsic Task Subspace for Few-Shot Learning via Prompt Tuning." IEEE/ACM TASLP 2024. DOI:10.1109/TASLP.2024.3430545 | 100 个 diverse NLP 任务可在**单一几百维子空间**内 reparameterize → ITS 普适性（已引，把这点讲透） |
| **N3** | R2-3.3 | Yan et al. "Contrastive Instruction Tuning." Findings of ACL 2024. arXiv:2402.11138 | near-OOD **hard negative** 提供比 cross-task negative 更有信息量的监督 → 承认 hard-neg 价值，同时界定 MELD-DS 剔除的是真噪声 |
| **N4** ⚠️ | R1-2, R2-2.4, AE-5 | Chi et al. "On the Representation Collapse of Sparse Mixture of Experts." NeurIPS 2022（arXiv 待核对，疑 2204.09179） | 形式化分析 sparse MoE **representation collapse**、随规模加剧 → 联合 top-k 路由不稳定 → 反衬 MELD 解耦 router 更稳 |
| **N5** ⚠️ | R1-2, R2-2.4, AE-5 | Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization..." ACL 2024（arXiv 待核对，疑 2401.06066） | 生产级 integrated MoE 靠 **shared expert + bias 负载均衡**对抗联合训练不稳定 → integrated 的"早层共享表示"优势伴随稳定性代价（tradeoff 叙事） |
| **N6** | R2-2.1 | "ClusterUCB: Efficient Gradient-Based Data Selection for Targeted Fine-Tuning of LLMs." arXiv:2506.10288 (2025) | **梯度相似→影响相似**做聚类 + bandit 预算分配；与 MELD-DS 最近邻 → 既验证我们的核心 intuition，又是差异化对象（我们=submodular 初始中心 + proxy + 多 GPU + 统一目标 + 误差界） |
| **N7** | R2-2.1 | Nikdan et al. "Influence Distillation: Efficient Data Selection at Scale via Influence Distillation." arXiv:2505.19051 (2025) | **landmark 精确 IF + 传播**；定位 batch/cluster-level IF 相关工作群 |
| **N8** | R2-2.1 | Axiotis et al. "Data-efficient learning via clustering-based sensitivity sampling: Foundation models and beyond." ICML 2024 | **clustering-based sensitivity sampling** 带可证明保证 → batch-level IF 非全新，凸显我们的组合贡献 |

**文献支撑落到三个核心 rebuttal 论点**：
1. **ITS 跨域**（R1-1/AE-1）：N1+N2 把"ITS 普适性"从我们的 t-SNE 上升为跨域文献共识，DP 是其表格实例化 → "test or discuss" 的 discuss 端有硬支撑。
2. **解耦 router 更稳**（R1-2/R2-2.4/AE-5）：N4+N5 把"联合训练 MoE 的不稳定"做实，于是 MELD"先训专家、再训固定专家上的 router"被论证为**消除 co-adaptation 不稳定的设计**，把 reviewer 的"质疑"转成"我们的优点"。
3. **batch-IF 定位**（R2-2.1）：N6+N7+N8 主动承认 cluster/batch-level IF 的相关工作群，把新颖性聚焦到"组合 + 理论保证 + 多 GPU 实现"，避免被 reviewer 抓"batch-IF 不新"。

---

## 8. 给群里的一句话总结

> AE + 2 位 reviewer 都给 Major（非 Reject），核心方法没被质疑。11 周足够：写作澄清约 1 天可清（9 条 A 类，多数是 1~2 句定义/解释）；零 GPU 的聚类开销核算 + failure case 用现有数据解决；真正要跑的只有 4 个轻量实验——**dense 7B 基线（P0，AE 点名）、Theorem 6 敏感性（P0）、noisy 鲁棒性、task-count 可扩展性**，约 2~3 周。文献已 survey 好 8 篇（N1–N8），把"ITS 跨域""解耦 router 更稳""batch-IF 定位"三处说法都补上了硬支撑，其中**解耦 router 那条能把 reviewer 的质疑直接转成我们的卖点**。请先看 §6 的 7 个决策点和 §9 的 5 个待确认项。

---

## 9. 需要向你 double check 的项（请逐条回我）

- **Q-A（最关键）**：Table I 的 **"LLM Baseline" 列到底是什么**？是"按任务族选用的最强 per-task LLM 基线（JellyFish/TableLLaMa/ExtractGPT）"，还是"单个在全任务上联合微调的模型"？这决定 R2-2.3 描述段定稿 + Table I 脚注，也决定它与新 dense 7B 列怎么并排。
- **Q-B（硬件）**：这轮 revision 实验用哪套？论文里写的是 **4×RTX 3090 + 2×A800**；我记忆里你常用 **2×RTX 4090**。这影响 C-①②③④ 的并行度和我排的 2~3 周窗口是否现实。
- **Q-C（full version [30] 现状）**：arXiv full version [30] 里**已经有哪些**额外实验/证明？（正文说"GPT-4 对比、全部 theorem 证明"在里面）我想知道哪些能直接接住溢出内容（D5 的 (c) 档），避免重复跑。
- **Q-D（计时日志）**：现有 run 是否留有**分阶段 wall-clock**（聚类 / 选择 / 训练 分别多少秒）？有 → R2-2.1 聚类开销直接从日志补；没有 → 需单次重计时（秒级，无所谓）。
- **Q-E（self-annotation 实现）**：EM→DI 这类 transform 确认是**纯自动的固定属性掩码**、无逐任务手工规则吧？（我按"自动"写了 R2-3.1 草稿，确认下以免 reviewer 反咬。）

**关于 LaTeX 源码**：是的，**请把论文的 LaTeX zip 发我**。本份会议纪要靠 PDF 已足够，但下一步要做的事都依赖源码：
1. **精确 label 引用**——response letter 要逐条写"见 §X.Y 第 Z 段 / Eq.(n) / Table m"，需要你们实际的 `\label`；
2. **页数审计 + §VII 压缩**（D5）——要在源码里量当前行占用、标出拟删句给你确认（按你"最小定向编辑"的偏好）；
3. **直接落 A 类草稿 + N1–N8 的 bib**（含你的 `\ymy{}`/`\ymyrev{}` 标注习惯和异色高亮）；
4. **核对 Q-A/Q-D** 的真实配置（Table I 列定义、Table III 计时来源）。

收到 zip + Q-A~Q-E 的回复后，我可以直接出 **point-by-point response letter 全文初稿** + 在源码上落 A 类全部改动。
