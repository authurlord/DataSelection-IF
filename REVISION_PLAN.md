# TKDE-2025-09-3230 Major Revision — 最小化修改实验计划

> 基于对本 repo 代码与中间结果的离线分析（未占用 GPU）。
> 分支：`reproduce`。LoRA 权重已从 51.11 拉回 120 个。
> **ITS / router 相关代码在另一个 repo**，本计划中相关条目标记为「DEFER（他 repo）」并留空。

---

## 0. Reviewer 意见主索引（编号）

| ID | 来源 | 内容 |
|----|------|------|
| **AE1** | AE | ITS 假设在非表格领域的验证/讨论 |
| **AE2** | AE | Theorem 6 在 batch cohesion 差时的敏感性分析 |
| **AE3** | AE | batch-IF runtime 未计入 k-center clustering，需完整 cost breakdown |
| **AE4** | AE | LLM Baseline 描述不清；需 matched dense multi-task 7B 基线 |
| **AE5** | AE | routing instability / task-count scalability / standalone vs integrated MoE |
| **AE6** | AE | 自标注增强数据偏差；noisy augmentation 鲁棒性证据 |
| **R1.1** | R1 | = AE1（ITS 非表格） |
| **R1.2** | R1 | standalone router 额外复杂度与不稳定性 vs 标准 MoE |
| **R1.3** | R1 | = AE6（自标注/增强偏差与噪声鲁棒性） |
| **R1.4** | R1 | 任务数增长时 expert specialization / routing 质量 |
| **R1.5** | R1 | failure case 分析 |
| **R2.1** | R2 | 相对 LESS/DataInf 的新颖性；clustering 开销计入 runtime |
| **R2.2** | R2 | = AE2（Theorem 6 σ 假设敏感性） |
| **R2.3** | R2 | baseline 清晰度：matched dense 7B + "LLM Baseline" 配置说明 |
| **R2.4** | R2 | standalone vs integrated MoE tradeoff（closely related tasks） |
| **R2.S1** | R2 | Fig.1 EM→DI 转换是自动还是手工规则 |
| **R2.S2** | R2 | Eq.1 Information Bottleneck 在实践中如何操作化 |
| **R2.S3** | R2 | PPL ratio 阈值 f(x) 是 heuristic；hard negative（ratio>1）是否有益 |
| **R2.S4** | R2 | Table III：DataInf init 1200s vs MELD-DS 115s 具体优化了什么 |

---

## 1. 复用/重算分类说明

- **W（pure-writing）**：纯写作/澄清，0 计算。
- **C（CPU-only）**：只需 CPU 重跑（聚类、FL 子模、归一化、top-k、读 logits、制表），不占 GPU。
- **T（retrain）**：从已有 `train-select*.json` 直接重训 expert（不重跑 selection 流程）；占 7B GPU。
- **P（full-pipeline-rerun）**：需重算梯度/IF（grad/ 已被删，必须 GPU 重算），但**只在 0.5B proxy 上**，很便宜。

---

## 2. 实验清单（每条对应多个 reviewer ID）

### E1 — Cost breakdown 扩展（含聚类开销）+ DataInf 对比解释
- **覆盖：AE3, R2.1, R2.S4**
- **答什么**：在 Table III 增列/拆分 init→**clustering(batch-division)**→gradient→IF→selection 各阶段 wall-clock + VRAM；解释 DataInf 1200s vs MELD-DS 115s（proxy 0.5B warm-up 省去 full-7B 梯度存储/计算）。
- **现状**：`eval_result/time_dict_*.npy` **已把 `batch-division` 单独计时**（如 WebTable 聚类≈23.6s / 梯度≈475s / IF≈49s）。聚类是 sklearn KMeans+Agglomerative，**纯 CPU**（`Selection_Pipeline.py:247-253, 587-661`），可干净重测。
- **分类：C + W**（聚类重测 CPU；其余制表与解释为写作）
- **最小修改**：读已存 time_dict 制表 + 选 1-2 个数据集 CPU 重跑聚类取干净数；R2.S4 纯写作。
- **工作量：低（数天）**

### E2 — Theorem 6 敏感性分析（batch cohesion 差时）
- **覆盖：AE2, R2.2（并强化 R2.1 novelty）**
- **答什么**：构造不同 batch cohesion：①k-center vs random clustering ②向 cluster 注入不同比例异质/噪声样本。测量 (a) within-batch 梯度 σ、(b) batch-IF 相对 per-sample IF 的近似误差、(c) 下游选择质量曲线。
- **现状**：聚类变体=CPU（复用 `cluster_vectors()` / `do_clustering()`）。但 **grad/ 已删 → 必须重算梯度**（`cal_IF_mp.py:284-320`，0.5B proxy GPU forward/backward）。iHVP/IF 数学是 CPU（`src/influence_batch.py`, `IF_Cal_mp.py`）。σ 计算无现成代码，需小增（复用 influence_batch 的逐样本梯度）。
- **分类：P**（仅 0.5B proxy，便宜）+ 少量 C
- **最小修改**：选 1 个语义内聚数据集（如 RE/RE，gradient 仅 ~133s）+ 1 个异质数据集；加 ~30 行 σ/误差统计；1 张新 figure。
- **工作量：中（约 1 周）**

### E3 — Matched dense multi-task 7B 基线
- **覆盖：AE4, R2.3**（注：用于隔离 router 收益的「对照」；router 侧分析在他 repo）
- **答什么**：同一 Mistral-7B，不走 MoE/router，把所有任务 `train-select.json` 合并联合微调一个 LoRA，逐任务 test→eval，加入 Table I。
- **现状**：可直接合并（总计约 31k 行，14 dataset）。训练 yaml 复用 `src/mistral-7B_template.yaml`；推理复用 `vllm_query_qwen.py`；评测复用 `src/evaluation.py`（按 (task,dataset) 分派，无需改）。
- **分类：T**（1 个 7B 训练 + 多 test 推理）
- **新增 glue**：合并脚本（~20 行）+ 逐任务评测循环（已有模式）。
- **工作量：中高**（单 7B 训练 ~数小时 + 全 test 推理；显存峰值 ~14G，单 3090 可行）

### E4 — Noisy augmentation 鲁棒性
- **覆盖：AE6, R1.3**
- **答什么**：向 `train-select.json` 的 `output` 注入 10/20/30% label noise，重训 expert，画性能降解曲线 vs baselines（证明 MELD-DS 选择对噪声的鲁棒性）。
- **现状**：数据格式 instruction/input/output（`Ablation_Study.py:132`）；eval 比对 predict vs ground-truth output（`src/evaluation.py:476-503`），无需改。最适合做干净噪声研究的分类型任务：**ER（match/mismatch）、DC、SM**。
- **分类：T**（多个小重训）
- **新增 glue**：噪声注入脚本（按任务翻转/随机化 output JSON 标签）。
- **工作量：中高**（建议限 2-3 dataset × 3 噪声档 ≈ 6-9 次小重训）

### E5 — Failure-case 分析
- **覆盖：R1.5**
- **答什么**：挑 main 方法最低分的 case 做错误归类。候选：**DC/rayyan（main F1≈0.84）、CTA/SimTab、CTA/WebTable**（CTA 用 Micro/Macro-F1，分数偏低）。
- **现状**：逐样本预测 CSV **已在** `output/Ablation/**/*.csv`（含 instruction/output/predict）；评测逻辑 `src/evaluation.py` 可逐行看 mismatch。
- **分类：W + C**（无需 GPU，直接读已存预测）
- **工作量：低**

### E6 — PPL-ratio f(x) / hard-negative 讨论（+可选小实验）
- **覆盖：R2.S3**
- **答什么**：承认 f(x) 是 heuristic；讨论 ratio>1（矛盾/hard negative）可能有益的场景，说明当前设计偏保守。
- **可选实验**：改 f(x) 阈值保留部分 ratio>1 样本 → 新 `train-select-w-ppl` 变体 → 重训对比。PPL 分数已存（`ppl/**/ppl-init-*.csv`），选择是 CPU；只有重训占 GPU。
- **分类：W**（主）/ **T**（可选实验）
- **工作量：低（写作）/ 中（可选）**

### E7 — Eq.1 IB 操作化说明
- **覆盖：R2.S2**
- **答什么**：说明 MI 不直接估计，实际用 iterative augmentation + contrastive surrogate 近似，IB 主要作概念指导。
- **分类：W** ｜ **工作量：低**

### E8 — Fig.1 EM→DI 转换自动化说明
- **覆盖：R2.S1**
- **答什么**：说明是基于规则的**自动**属性映射（如 DI 按 `ATTR_DICT` mask 指定属性，见 `README_dataset.md`），非逐例手工。
- **分类：W** ｜ **工作量：低**

### E9 — "LLM Baseline" / Table I 配置澄清
- **覆盖：AE4（描述部分）, R2.3（描述部分）**
- **答什么**：明确 Table I 的 LLM Baseline 具体配置（Jellyfish-13B / TableLLaMa-7B / ExtractGPT 按任务）。本 repo 未找到独立 baseline 训练脚本，需作者确认实际配置（可能在他 repo）。
- **分类：W**（需作者输入） ｜ **工作量：低**

---

## 3. DEFER（ITS / router，代码在他 repo — 本 plan 留空）

| 条目 | 覆盖 ID | 备注 |
|------|---------|------|
| ITS 非表格领域验证（t-SNE/subspace overlap） | AE1, R1.1 | 他 repo |
| Routing instability（confidence/entropy/variance） | AE5, R1.2 | 他 repo |
| Task-count scalability（3→5→7→10，expert utilization） | AE5, R1.4 | 他 repo |
| Standalone vs integrated MoE tradeoff（Table VII 扩展） | AE5, R2.4 | 他 repo（Mixtral 对比结果在 evaluation.ipynb） |

---

## 4. 实验 × Reviewer 覆盖矩阵

| 实验 | 分类 | AE1 | AE2 | AE3 | AE4 | AE5 | AE6 | R1.1 | R1.2 | R1.3 | R1.4 | R1.5 | R2.1 | R2.2 | R2.3 | R2.4 | R2.S1 | R2.S2 | R2.S3 | R2.S4 |
|------|------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| E1 cost breakdown | C+W | | | ✔ | | | | | | | | | ✔ | | | | | | | ✔ |
| E2 Thm6 sensitivity | P | | ✔ | | | | | | | | | | (✔) | ✔ | | | | | | |
| E3 dense 7B 基线 | T | | | | ✔ | | | | | | | | | | ✔ | | | | | |
| E4 noisy aug | T | | | | | | ✔ | | | ✔ | | | | | | | | | | |
| E5 failure case | W+C | | | | | | | | | | | ✔ | | | | | | | | |
| E6 PPL f(x) 讨论 | W/(T) | | | | | | | | | | | | | | | | | | ✔ | |
| E7 Eq.1 IB | W | | | | | | | | | | | | | | | | | ✔ | | |
| E8 Fig.1 自动化 | W | | | | | | | | | | | | | | | | ✔ | | | |
| E9 LLM Baseline 说明 | W | | | | ✔ | | | | | | | | | | ✔ | | | | | |
| DEFER (他 repo) | — | ✔ | | | | ✔ | | ✔ | ✔ | | ✔ | | | | | ✔ | | | | |

---

## 5. 磁盘上可复用的中间结果（已确认）

- ✅ `train/**/train-select*.json`（268 个）— 所有方法的选择结果，可直接重训
- ✅ `ppl/**/ppl-init-*.csv`（128）— PPL 分数
- ✅ `Influence_single/**/result_df.csv` — 单样本 IF 结果
- ✅ `eval_result/*.npy`（含 `time_dict_*` 的分阶段计时）— 最终指标 + cost 数据
- ✅ `output/Ablation/**/*.csv`（~206）— 逐样本预测
- ❌ `grad/`、`grad_single/`、`selection/**/*.pkl`、`Influence/**/*.pkl` — 已删（gitignore），重算 batch-IF 需 GPU（仅 0.5B proxy）
- ⚠️ 所有 base model 路径需从 `/data/home/wangys/...` 改为 `/ssd_data/models/...`

---

## 6. 建议执行顺序（按性价比）

1. **先做 0 GPU 的**：E5, E7, E8, E6(写作部分), E1(制表+R2.S4), E9 → 几天内可写进 response。
2. **再做便宜 GPU**：E2（仅 0.5B proxy）。
3. **最后做 7B 训练**：E3（AE 点名，优先级最高）、E4。
4. ITS/router（DEFER）在他 repo 推进。
