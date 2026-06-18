# MoE 重建 Backup Survey（router 源码丢失 → 重搭方案 + lora-merge 可行性）

> 状态(2026-06-15 更新):**改为执行** —— 用户授权在 **12.43(B 训练完成后,cuda6,7)实跑 multi-LoRA MoE**,**重点拿 ITS + 需要的趋势,可不符 MELD 原 claim,可自标注路由数据**。MELD 的 MoE 本质 = standalone router + multi-LoRA vLLM 服务 + LoRA merge。router 源码丢失 → 按本文路线重建。

## 执行计划（优先级:ITS > scalability/routing > lora-merge）

> 资源:12.43 cuda6,7 等 B(dense-7B)训完(~3.5h)释放后用;51.11 V100 跑 C 期间不可占。

**P0 — ITS 证据(AE-1, R1-1)**:已有 RE/ER/CTA 的逐样本 proxy 梯度(Exp A 产物,51.11)。
- 每任务取梯度均值=task vector → 跨任务 subspace overlap(principal angle)+ t-SNE。
- 非表格点:对 `datasets/`(math_with_reason、grammars、events_classification_biotech、GLUE/mrpc)各取子集,过 0.5B proxy 抽 task vector(轻量 GPU)→ 叠加 t-SNE,验证「非表格任务落在同一 ITS 的可分簇」。
- 产出:t-SNE 图(表格+非表格)+ subspace overlap 矩阵。**CPU + 轻量 GPU,B 训完即可在 12.43 或 51.11 做。**

**P1 — multi-LoRA MoE + 路由趋势(R1-4, AE-5, R1-2)**:
- 自标注路由数据(复用 `train/router/train.csv` + 各任务样本)→ 训练分类式 router(bge/ModernBERT,轻量)。
- 任务数 n∈{3,5,7,10}:路由命中率 + expert 利用率(负载熵)随 n 的曲线。
- 路由稳定性:sigmoid 置信度 / entropy。
- multi-LoRA 服务:复用 `router/vllm_MoE_query.py` + vLLM LoRARequest 动态挂载 top-m。

**P2 — lora-merge 可行性(R2-4)**:见下 §3 的 V1–V3。
> 现存可复用资产:`train/router/train.csv`(3600 行,7 任务类/18 dataset 标注)、每任务的 expert LoRA 权重(已拉回 120 个)、`router/` 下的 `vllm_MoE_query.py`/`vllm_router_query.py`/`vllm_MoE_query_optimize.py`(推理外壳尚在,缺训好的 router 模型与训练脚本)。

## 1. 系统三要素与丢失部分

| 组件 | 作用 | 现状 |
|---|---|---|
| **Standalone router** | 把 query 分派到 top-m experts | ❌ 训练脚本+权重丢失;`train/router/train.csv` 在 |
| **Multi-LoRA vLLM 服务** | 一个 base + 多 LoRA 动态加载/批处理 | ⚠️ 外壳 `router/vllm_MoE_query.py` 在;vllm 原生支持 multi-LoRA(LoRARequest) |
| **LoRA merge** | 把 top-m experts 合成一个适配器(或加权激活) | ⚠️ `add_weighted_adapter` 路径在 `vllm_MoE_query.py` 中被引用,需验证 |

## 2. 重建 Router —— 可选方案(从已有工作)

要点:MELD 的卖点是**解耦**(先训好固定 experts,再训 router),所以 router 只是一个在固定 expert 集上的分派器,重建相对独立。

1. **分类式 router(最贴合原设计,推荐)**:用编码器(ModernBERT / bge-large-en)做 multi-label 分类,标签=expert(task/dataset)。训练数据 `train/router/train.csv` 现成。早期 survey 显示原 router 即 ModernBERT+LoRA 多标签头(P/R≈0.92/0.90)。重建成本低(~分钟级训练)。
2. **Embedding 最近邻 router(无训练)**:用 RAG 编码器对每个 expert 的训练样本求质心,query 取 top-m 最近质心。零训练、可解释,作为 router 的下界基线。
3. **Arrow / gradient-free routing**(2024):用每个 LoRA 的右奇异向量做原型,query 投影选 expert,无需 router 训练。适合 plug-and-play 论述。
4. **对比式 router(原文用 contrastive loss)**:共享 RAG 编码层 + 对比目标,把 query 拉近其 expert 簇。与 Theorem 3 的簇可分性呼应。

**建议**:先做 (1) 复刻原结果 + (2) 作下界,再补 (3) 作 plug-and-play 论据。

## 3. LoRA Merge / 组合 —— 可行性与方案

reviewer 关心点 = 多 expert 如何合成。已有成熟做法:

| 方法 | 机制 | 工具 | 适用 |
|---|---|---|---|
| **加权线性合并** | ΔW = Σ wᵢ BᵢAᵢ | PEFT `add_weighted_adapter(combination_type="linear")` | top-m 合一,推理最快 |
| **cat 合并** | 拼接 LoRA(秩相加) | PEFT `combination_type="cat"` | 保留各 expert 子空间,秩增大 |
| **TIES / DARE** | 裁剪+符号对齐/随机丢弃再合并,缓解干扰 | PEFT `ties`/`dare_linear` | expert 冲突大时 |
| **LoRAHub** | 黑盒优化少量样本上的组合系数 | LoRAHub | few-shot 自适应权重 |
| **multi-LoRA 不合并(动态激活)** | base + 每请求挂 top-m LoRA | vLLM LoRARequest / S-LoRA / Punica | 服务态,免合并 |

**可行性验证计划(最小,后续执行)**:
- V1:取 2-3 个相关 expert(如 DC/beer+hospital+rayyan),用 `add_weighted_adapter` 等权合并 → 在各自 test 上评测,对比「单 expert」与「合并」F1。验证合并不显著掉点。
- V2:跨域 expert(EM vs CTA)合并 → 预期掉点(子空间冲突)→ 佐证「解耦+top-m 动态激活」优于「全合并」,支撑 standalone 设计论述。
- V3:vLLM multi-LoRA 动态激活 top-m vs 合并:对比延迟/吞吐/精度,给 R2-2.4 的 tradeoff 提供数据。
- 指标:per-task F1、合并前后差、推理延迟/显存。

## 4. 重建里程碑(待 source 或自建,均不阻塞当前非-MoE 实验)

1. router 训练脚本(方案1)+ 用 `train/router/train.csv` 训练 → 复刻路由命中率。
2. 路由→top-m expert→(合并 or 动态激活)→ vLLM 推理(复用 `router/vllm_MoE_query.py` 外壳)。
3. lora-merge 可行性 V1–V3。
4. 接 D(task-count scalability)、AE-5(routing 稳定性)、R2-2.4(standalone vs integrated)经验实验。

## 5. 关键参考(供落 bib / 深读)
- PEFT `add_weighted_adapter`(linear/cat/ties/dare)— LoRA 合并标准实现。
- TIES-Merging(Yadav 2023)、DARE(Yu 2024)— 缓解合并干扰。
- LoRAHub(Huang 2024)— few-shot LoRA 组合。
- S-LoRA(2023)、Punica(2023)、vLLM multi-LoRA — 多 LoRA 高效服务。
- Arrow routing / Phatgoose(2024)— gradient-free LoRA 路由。
- (原文已引)Mixtral、DeepSeekMoE — integrated MoE 对照。

> 待用户提供原 router repo 后,优先复用;若确认丢失,按本路线方案1+方案2 重搭(成本低)。
