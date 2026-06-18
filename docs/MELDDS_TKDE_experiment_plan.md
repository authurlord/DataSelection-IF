# MELD-DS TKDE Revision — 实验执行表（供你跑实验）

> 用途：5 个新实验的可执行规格，**从难到易**排列。每个给：目标 / setting（分步）/ 数据集 / 模型 / 输出物 / fallback。
> 硬件基线：4×RTX 3090（24G）主实验；2×A800（80G）仅用于需要大显存的参考跑（如 7B 全参/per-sample IF 参考），**论文里速度按 3090 报**。
> 默认沿用论文配置：proxy = Qwen-2.5-0.5B；expert backbone = Mistral-7B + LoRA；RAG backbone = bge-large-en；c=30%，|B|=8，k=3，σ_iter=3，|D_i|=8，τ=0.02；每实验跑 3 次取平均。

---

## 速览表

| # | 实验 | 难度 | 回应意见 | 数据集 | 模型 | 核心输出 | 预计 |
|---|---|---|---|---|---|---|---|
| **A** | Theorem 6 batch-cohesion 敏感性 | ★★★★★ | AE-2, R2-2.2 (+ 修 Thm 6) | WebTable, RE(WikiTable), Amazon-Google | 0.5B proxy（IF/参考）+ 7B expert（F1） | σ–误差曲线、\|B\|-sweep U 型、random vs k-center 表 | 2–3d |
| **B** | matched dense multi-task 7B | ★★★★☆ | AE-4, R2-2.3 (+ 喂 R2-2.4) | 全 19 数据集 / 10 任务 | Mistral-7B + LoRA（单模型，无 MoE/router） | Table I 新列 | 2–3d |
| **C** | noisy augmentation 鲁棒性 | ★★★☆☆ | AE-6, R1-3 (+ 喂 R2-3.3) | Amazon-Google, WebTable, Rayyan | 0.5B proxy + 7B expert | F1–噪声率降解曲线（4 方法） | 2–3d |
| **D** | task-count 可扩展性 | ★★☆☆☆ | R1-4, AE-5 | 10 个低重叠任务（嵌套子集） | 已训 experts + 重训 router | per-task F1/利用率/命中率 vs n | 1–2d |
| **E** | ITS 非表格点 | ★☆☆☆☆ | R1-1, AE-1 | +1 个文本类任务 + 现有任务 | Figure 2 同款 task-vector 模型 | Figure 2 叠加新簇 | 0.5–1d |

> 关键提醒：实验 A 同时是**修 Theorem 6 的经验依据**（见证明审计文档）——σ(|B|) 随 |B| 增大是把"√|B| bug"改成"sweet-spot 解释"的关键证据。优先级最高的是 A 和 B。

---

## A. Theorem 6 batch-cohesion 敏感性 ★★★★★（最难）

**为什么最难**：需要先算一份**逐样本 IF 参考值**（这正是 batch-IF 要避免的昂贵计算），才能量化 batch-IF 的近似误差；还要插桩抽取 batch 内梯度统计量 σ，并跑多组退化设置。

**目标**：量化 batch-IF 近似误差与下游 F1 随 batch cohesion（即 σ）退化的变化；坐实 Theorem 6 的 σ 假设，并为定理修复提供经验曲线。

**Setting（分三组条件）**：
1. **参考值**：在一个可控规模的 pool（建议每任务取 6k，与 DataInf 的 6k 对齐）上，用 **0.5B proxy** 计算**逐样本 IF**（Eq.8），作为 batch-IF 误差的 ground-truth。注意：参考用 proxy 而非 7B，以保可计算性（7B 逐样本 IF 不可行，这本就是动机）。
2. **条件 1 — 聚类质量**：固定 |B|=8，对比 (a) k-center 聚类（默认，S'_init 作初始中心）vs (b) random 聚类。测：每样本 |I(z) − I(B)| 的均值（近似误差）、最终 F1。
3. **条件 2 — σ 注入**：向每个簇按 {0,10,20,30%} 比例注入**跨任务/跨域样本**（人为抬高簇内 σ）。测：近似误差、F1，以及实测 σ。
4. **条件 3 — |B| sweep + σ 测量**：|B| ∈ {1,4,8,16,32}（扩展现有 Table V）。**在每个 |B| 记录实测 batch 内梯度标准差 σ(|B|) 和近似误差**。这一组直接检验误差是按 σ(|B|)/√|B|（U 型，与实测一致）还是按论文写的 1/√|B|（单调下降，与 Table V 矛盾）。

**数据集**：WebTable（CTA，14856，与 Table V 同）、WikiTable（RE，6369）、Amazon-Google（ER，6612）——全部沿用 Table V 已有行，便于对照。

**模型**：IF/梯度用 Qwen-2.5-0.5B proxy；最终 F1 用 Mistral-7B expert。

**输出物**：
- 图 1：实测 σ vs 近似误差（验证误差随 σ 线性上升，匹配 Theorem 6 的 O(σ·‖H⁻¹‖)）。
- 图 2：**σ(|B|) vs |B|**（展示 σ 随 |B| 增大——这是修定理的核心证据）+ 同图叠 F1 vs |B|（复现 Table V 的 U 型 sweet-spot）。
- 表：k-center vs random 在 |B|=8 的"误差/F1"差（证明语义聚类是降 σ 的机制，非假设）。

**Fallback**：
- 参考逐样本 IF 仍太贵 → 每任务降到 2k 子集做参考；或用**梯度余弦方差**作 σ 代理，跳过完整逐样本 IF。
- σ 注入实现繁琐 → 只保留"random vs k-center + |B| sweep"两组，这两组已足够证明机制和 sweet-spot。
- 只想要最小回应 → 仅做条件 3（|B| sweep + σ 测量），一张 U 型图 + σ(|B|) 曲线即可同时回应 R2-2.2 和修 Theorem 6。

---

## B. matched dense multi-task 7B 基线 ★★★★☆（次难，算力重）

**为什么难**：要把一个 Mistral-7B 在全部 10 任务上联合 SFT，并在 19 个数据集全量评估，算力是主要成本（但 pipeline 标准）。

**目标**：隔离"MoE 结构 + standalone router"的收益（AE 点名）；给 Table I 加干净基线；为 R2-2.4 提供 integrated-style 的 dense 参考点。

**Setting**：
1. backbone 用 **Mistral-7B**（与 MELD experts 同源，才叫 matched）。
2. **单个 LoRA** 在**全部 10 任务的训练数据并集**上联合 SFT，**不走 MoE、不用 standalone router**。训练数据用与 MELD 相同的 per-task 训练集（保证除"MoE/router"外其它条件一致）。
3. 在全部 19 数据集用相同 metrics 评估，加入 **Table I 新列 "Dense-MT 7B"**。
4. ⚠️ **务必区分**：此基线 ≠ 现有 Table VII 的 "MELD w/o MoE"（后者是**每任务一个单专家**；此处是**一个模型跨全部任务**）。response letter R2-2.3 已点明二者区别。

**数据集**：全部 19 数据集 / 10 任务（覆盖 Table I 全行）。

**模型**：Mistral-7B + LoRA（单一、多任务）。

**输出物**：Table I 新增一列。预期 MELD（多稀疏专家）> Dense-MT 7B（单密集模型），坐实 Theorem 2 的"多稀疏胜一密集"。数值填 response letter R2-2.3 的 `[TBD]`。

**Fallback**：
- 全 19 数据集评估太久 → 先报 Table II 的 13 个数据集子集，其余进 full version。
- 公平性：LoRA rank / 训练步数预算与 MELD 单专家训练**对齐**（或按总算力匹配），避免被质疑配差。
- 若 Dense-MT 7B 因任务干扰而明显劣化——这本就是预期的正向结果，无需 fallback，但要在 caption 说明训练预算匹配。

---

## C. noisy augmentation 鲁棒性 ★★★☆☆（中）

**目标**：证明在噪声/偏差增强下 MELD-DS 的 quality+complexity 度量能抗噪（R1-3, AE-6）；顺带支撑 R2-3.3（contradictory 样本）。

**Setting**：
1. 向 X_aug 注入标签噪声，比例 {0,10,20,30%}（随机翻转/打乱该比例的 self-annotation 标签）。
2. 在 c=30% 下对比：**MELD-DS** vs (a) full-pool 不选择 vs (b) DSIR（纯 diversity）vs (c) SuperFilter（纯 quality）。
3. 测每个噪声率下的下游 F1 → 降解曲线。

**数据集**：3 个不同结构的代表任务：Amazon-Google（ER）、WebTable（CTA）、Rayyan（DC）。

**模型**：选择阶段用 0.5B proxy；SFT/评估用 Mistral-7B expert。

**输出物**：F1–噪声率降解曲线（MELD-DS + 3 baseline × 3 数据集）。预期 MELD-DS 降解最慢。填 R1-3 的 `[TBD]`。

**Fallback**：
- 3×4×4 组合太多 → 降到 2 数据集（Amazon-Google + WebTable）× {0,20%} × {MELD-DS, full, DSIR}；关键看斜率。
- 最小版 → 仅 MELD-DS vs full-pool（最干净的鲁棒性陈述）。

---

## D. task-count 可扩展性 ★★☆☆☆（中偏易）

**目标**：展示任务数 n 增长时 per-task 性能稳定、专家利用率不塌缩、路由质量维持（R1-4, AE-5）。

**Setting**：
1. **复用已训练的 experts**（不重训专家）。
2. n ∈ {3,5,7,10}：按**固定顺序嵌套**纳入任务（如 n=3 用 EM/CTA/DC，n=5 加 ED/DI，依此类推），**只重训 standalone router**，评估：per-task F1、路由命中率、专家利用率（负载分布熵 / 最大-最小负载差）。
3. （可选）同 n 下对比 Mixtral router。

**数据集**：从 19 数据集里选 10 个低域重叠任务做嵌套序列（参照现有 6-dataset C-D/C-T 选法扩到 10）。

**模型**：已训 experts（Mistral-7B LoRA）+ 重训 router。

**输出物**：(i) per-task F1 vs n（稳定性）；(ii) 利用率熵/负载差 vs n（不塌缩）；(iii) 命中率 vs n。预期 per-task F1 稳定、负载均衡。填 R1-4 的 `[TBD]`。

**Fallback**：
- 每个 n 重训 router 太贵 → 用**同一个已训 router**、评估时 mask 到 n 任务（更便宜的近似）。
- 再省 → 只报 n ∈ {3,6,10} 三点。

---

## E. ITS 非表格点 ★☆☆☆☆（最易）

**目标**：证明 ITS 不限于表格（R1-1, AE-1 的 "test" 端）。

**Setting**：
1. 选 1 个结构差异更大/更偏自由文本的任务（如自由文本 RE，或一个短文本分类任务；避免与现有 t-SNE 的表格任务重叠）。
2. 用与 Figure 2 同款方法（Hendel/Qin 的 task-vector 抽取）算其 task vector。
3. 叠加到现有 Figure 2 的 t-SNE，检查它是否在同一子空间里形成**可分的独立簇**。

**数据集**：1 个额外的非表格/文本任务 + 现有任务（做 t-SNE）。

**模型**：与 Figure 2 现有 t-SNE 同款（生成 task vector 的 LLM）。

**输出物**：更新后的 Figure 2（含非表格任务簇）+ 正文 2–3 句。填 R1-1 的 "test" 部分。

**Fallback**（这是最容易砍的一个，对应 D1 决策）：
- 直接用 Figure 2 里已有的文本类任务（RE、EL）作为"结构不同"证据 + 文献论证（Aghajanyan 2021、Qin TASLP 2024），写进 Limitation，**不跑新实验**。

---

## 执行顺序建议

1. **先并行启动 A 和 B**（都是 P0，A 插桩 + 参考跑、B 联合训练，互不抢占可以并行；B 占 GPU 多，A 占 proxy/工程）。
2. **A 跑完立刻出 σ(|B|) 曲线** → 同步交给证明审计文档去修 Theorem 6（两件事互相印证）。
3. **再做 C 和 D**（中等，可串行或错峰）。
4. **E 看 D1 决策**：做就最后补 Figure 2；不做就纯文献 + Limitation。
5. 全部 `[TBD]` 数值回填 response letter 对应位置。

> 分阶段 wall-clock（聚类/选择/训练）你已说有日志可抽 → 那个不在本表（属 B 类零新 GPU），抽出来直接填 response letter 的聚类开销 `[TBD]` 即可。
