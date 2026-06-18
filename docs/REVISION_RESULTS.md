# MELD-DS Revision — 实验结果汇总（供判断）

> 滚动更新。所有结果在 51.11(V100/fp16, deepspeed env)产出,代码隔离在 `revision_exp/`,未改 main。
> proxy=Qwen2.5-0.5B；每数据集取 pool 前 3000 样本算逐样本梯度作参考。

---

## Exp A — Theorem 6 batch-cohesion 敏感性（AE-2, R2-2.2；并修 Theorem 6）

度量:batch-IF 用逐样本 IF(|B|=1)作真值;误差=两者 z-score 后的平均绝对差(method=proposed/DataInf);σ_norm=簇内梯度标准差/全局梯度RMS。

### RE/RE（pool=6657,取3000）✅
**(1) error vs σ(固定|B|=8,按比例 p 打乱语义簇抬高 σ)** —— 单调,坐实 Theorem 6 的 σ 依赖:

| p(打乱比例) | σ_norm | error |
|---|---|---|
| 0.00 | 0.850 | 0.590 |
| 0.10 | 0.868 | 0.613 |
| 0.25 | 0.888 | 0.694 |
| 0.50 | 0.912 | 0.745 |
| 1.00 | 0.934 | 0.818 |

**(2) |B| sweep(语义聚类):σ(|B|) 单调上升** —— 修 Theorem 6 的核心证据(σ 非恒定):

| \|B\| | σ_norm | error |
|---|---|---|
| 1 | 0.000 | 0.000 |
| 4 | 0.800 | 0.497 |
| 8 | 0.850 | 0.590 |
| 16 | 0.881 | 0.567 |
| 32 | 0.907 | 0.584 |

**(3) random vs 语义聚类(|B|=8)** —— 语义聚类同时降低 σ 和误差(机制成立):

| | σ_norm | error |
|---|---|---|
| random | 0.934 | 0.837 |
| semantic | 0.850 | 0.590 |

图:`revision_exp/results/RE/fig1_error_vs_sigma.png`、`fig2_Bsweep.png`

### ER/amazon-google（pool=6612,取3000）✅ —— 最干净
| 度量 | 值 |
|---|---|
| error vs σ(\|B\|=8,p扫) | σ 0.002164→0.003164,err **0.225→0.304→0.425→0.592→0.771**(单调) |
| B_sweep σ(\|B\|) | 1:0 → 4:0.560 → 8:0.633 → 16:0.731 → 32:0.824(单调上升) |
| B_sweep error | 0 → 0.191 → 0.225 → 0.282 → 0.377(单调上升) |
| random vs semantic(\|B\|=8) | σ 0.929/0.633,err **0.798/0.225**(差 3.5×) |

### CTA/WebTable（pool=14856,取3000）✅
| 度量 | 值 |
|---|---|
| error vs σ(\|B\|=8,p扫) | σ 0.858→0.932,err **0.468→0.533→0.612→0.75→0.871**(单调) |
| B_sweep σ(\|B\|) | 1:0 → 4:0.818 → 8:0.858 → 16:0.886 → 32:0.916(单调上升) |
| B_sweep error | 0 → 0.499 → 0.468 → 0.491 → 0.561 |
| random vs semantic(\|B\|=8) | σ 0.931/0.858,err **0.875/0.468** |

### 跨数据集一致性（Exp A 核心结论）
**三个数据集(RE/ER/CTA)趋势一致**:(1) 固定 \|B\| 下 error 随 σ 单调上升;(2) σ(\|B\|) 随 \|B\| 单调上升;(3) 语义聚类比 random 显著降低 σ 和误差。→ 坐实 Theorem 6 的 σ 依赖,并证明「σ 恒定」假设不成立(σ 随 \|B\| 增长),解释 Table V 的 U 型 sweet-spot。

**RE 结论**:三组趋势全部成立。最强证据是 (1) 固定 |B| 下 error 随 σ 单调上升,直接验证 Theorem 6 的 σ 项;(2) σ(|B|) 随 |B| 增大,说明论文「σ 恒定→误差 1/√|B| 单调下降」的读法不成立,真实是 σ(|B|)/√|B| 的权衡(对应 Table V 的 U 型 sweet-spot)。

---

## E1 — Cost breakdown / 聚类开销（AE-3, R2-2.1, R2-S4）✅

从现有 `eval_result/time_dict_*.npy` 抽分阶段 wall-clock(秒)。**聚类(batch-division)仅占总选择时间 1.8–4.3%**,梯度计算占 75–85%:

| 数据集(run) | 聚类 | 梯度 | IF | init | 总计 | 聚类占比 |
|---|---|---|---|---|---|---|
| CTA SimTab | 6.2 | 302.9 | 27.8 | 15.4 | 352.3 | 1.8% |
| CTA WebTable | 23.7 | 475.5 | 48.6 | 23.1 | 570.9 | 4.1% |
| ER amazon-google (|B|=4) | 5.7 | 130.2 | 47.8 | 12.2 | 195.9 | 2.9% |
| RE RE (|B|=4) | 6.1 | 120.6 | 42.6 | 6.8 | 176.2 | 3.5% |

**回填 R2-2.1**:WebTable(14856 样本)聚类 ~23.7s,占 570.9s 选择时间的 **4.1%** → 聚类开销不可忽略性被排除,3-10% runtime 声明在计入聚类后仍成立。
**回填 R2-S4**:DataInf 1200s init = full-7B 逐样本梯度计算+存储(22GB);MELD-DS 115s = 0.5B proxy warm-up + 批级梯度(3.4GB)。差异来自 proxy 规模 + 批级,而非省略 IF 步骤。

---

## E5 — Failure-case 分析（R1-5）✅（本地 CPU,基于已有预测 CSV）

| 失败案例 | 指标 | 错误分解 | 失败模式 |
|---|---|---|---|
| ER/semi-text-c | F1=0.301 | 正例仅13.9%; TP225/FP689/FN354/TN2911; 无格式错 | **类不平衡+高假阳**:难的半结构化 computer 匹配上过度判 match(precision 0.246) |
| AVE/oa_mine | exact=0.763 | 空预测仅0.2% | **抽取值不精确**(格式/释义差异,非漏抽) |
| CTA/SimTab | micro=0.768 | 常见类0.808 vs 稀有类0.551 | **稀有类型识别差**→ 拉低 macro-F1(0.734) |

→ 回填 R1-5:MELD 在闭域/结构受限任务强,在(a)极端类不平衡的半结构化匹配、(b)开放域细粒度抽取、(c)长尾稀有类型上相对弱。与 §VI-B 现有论述一致,补成 appendix 小节。

## C — Noisy augmentation 鲁棒性（AE-6, R1-3）🔄

**ER/amazon-google,MELD-DS-selected 训练(1984行)在 {0,10,20,30}% label noise 下的 test F1:**
| noise | 0% | 10% | 20% | 30% |
|---|---|---|---|---|
| **MELD-DS-selected** | 0.765 | 0.779 | 0.768 | 0.652 |

**full-pool 对照(6612行,3×数据):** 0%=0.834, 10%=0.835, 20%=0.791, 30%=0.779。

⚠️ **设计修正(诚实记录)**:full-pool 比 MELD-DS-selected 更鲁棒(数据量 3× → 噪声被稀释)。这**不是公平对比**(full 用 3× 算力)。MELD-DS 的卖点是「30% 预算」,所以正确对比是**同等 30% 预算下 MELD-DS vs random/DSIR/SuperFilter**。
**同 30% 预算对比(test F1):**
| noise | MELD-DS | random-30% | full-pool(100%) |
|---|---|---|---|
| 0% | 0.765 | 0.837 | 0.834 |
| 10% | 0.779 | 0.796 | 0.835 |
| 20% | 0.768 | 0.717 | 0.791 |
| 30% | 0.652 | 0.605 | 0.779 |
| **相对降解** | **14.8%** | **27.7%** | 6.6% |

**⚠️ MIXED 结果(诚实)**:
- ✅ 斜率支持:MELD-DS 降解 ~2× 慢于 random(14.8% vs 27.7%),≥20% 噪声时反超 random → 「MELD-DS 选的数据更抗噪」成立。
- ❌ 异常:0% 时 random(0.837)> MELD-DS(0.765)→ 单次方差(无 seed)或 train-select 非最优变体所致;削弱「MELD-DS 干净时也更好」。
- ⚠️ 框架问题:本实验是「向已选集注噪」,测的是 selected-set 训练鲁棒性,**非**「选择时过滤噪声」(claim 的真正机制)。
**✅✅ 过滤机制实验(正确框架,CLEAN headline)—— 向 pool 注 30% 噪声,用 0.5B proxy+init adapter 算 per-sample ppl:**
| | flipped(噪声) | clean | 
|---|---|---|
| ppl 均值 | **1.719** | 1.015(高 1.69×) |

- 去掉 ppl 最高的 30%(质量分过滤)→ 其中 **91.1% 是噪声样本**(基率 29%)
- 该操作**捕获 94.3% 的注入噪声**
- → **MELD-DS 的质量分(ppl)几乎完美识别并过滤标签噪声**,直接坐实 R1-3/AE-6 的机制 claim。
- 图:`revision_exp/results/figs/expC_ppl_filter.png`(flipped vs clean ppl 分布几乎不重叠)。

**C 小结**:headline = 质量分过滤噪声(94% recall);辅以训练降解(MELD-DS 降解 ~2× 慢于 random)。训练降解的 0% 异常用多 seed 收尾(可选)。
- ⏳ CTA/WebTable 待训(可选,ER 已足够支撑)。

## ITS — 内在任务子空间证据（AE-1, R1-1）✅（表格部分,CPU）

洞察:每个任务的 qwen-0.5B init LoRA adapter = 该任务的 task vector。16 个异构 DP 任务:
- 参数维度 688,128,但 task vector 的**内在维度仅 8(90% 方差)/ 11(95%)**。
- 累计方差:PC1=34.6%, PC1-2=52.2%, PC1-6=85.4%。
- → 异构 DP 任务的适配被压缩在 ≤8 维**共享低维子空间**,坐实 ITS 假设。
- 图:`revision_exp/results/ITS/its_tsne.png`(任务聚类)、`its_varexp.png`(内在维度曲线)。
- **非表格点(AE-1 "test")✅**:在 grammar(字符串变换,结构与表格 DP 迥异)上训同配置 qwen-0.5B LoRA → 加入后 **17 任务内在维度仍为 8**(不增维);grammar 对表格任务平均余弦 0.945(vs 表格内 0.970)→ 略可区分但**共嵌于同一子空间**,非离群。坐实「ITS 不限于表格」。图:`revision_exp/results/ITS_v2/its_tsne.png`。
  - (math 任务因 prompt 超 cutoff 被滤,失败;grammar 已足够。可选再补。)

## MoE — 重建(router 源码丢失)在 12.43 实跑

### ✅ Router scalability（R1-4, AE-5）—— bge-m3 嵌入 + 线性分类 router,dataset 级 experts
| # experts n | top-1 | top-2 | 利用率均衡 | 置信度 | 决策熵 |
|---|---|---|---|---|---|
| 3 | 1.000 | 1.000 | 1.000 | 0.966 | 0.142 |
| 6 | 1.000 | 1.000 | 1.000 | 0.969 | 0.150 |
| 9 | 0.975 | 1.000 | 0.999 | 0.940 | 0.241 |
| 12 | 0.981 | 1.000 | 0.999 | 0.936 | 0.269 |
| 15 | 0.980 | 1.000 | 0.998 | 0.933 | 0.298 |
| 18 | 0.951 | 0.976 | 0.995 | 0.911 | 0.363 |

→ **experts 3→18:top-1 仅 1.0→0.951,top-2≥0.976,利用率均衡 ~1.0(无塌缩)** → 标准 standalone router 随任务数增长仍保持高路由质量+均衡利用率。回应 R1-4/AE-5。图:`revision_exp/results/figs/router_scalability.png`。

### LoRA merge 冲突验证（用户指令:merge 不能冲突)—— 关键发现
文献(deep-research)正解:conflict-free = **dare_ties**(DARE drop+rescale → TIES trim/elect-sign/disjoint),或**不合并、vLLM per-request 服务**。
**验证(4 异构专家 beer/amazon/re/ave merge → 各任务 F1 vs 单 adapter):**
| 任务 | 单 adapter | merged d=0.5 | merged d=0.2 |
|---|---|---|---|
| RE/RE(micro) | 0.772 | 0.774 ✓ | 0.750 |
| DC/beer | 0.993 | 0.856 ✗ | 0.530 ✗ |
| AVE | 0.762 | 0.659 ✗ | 0.615 ✗ |

→ **异构专家权重合并必然冲突**(各 density 都掉,RE 因主导幸存);更激进 DARE drop 反更差(delta 不够冗余)。与文献一致:**merge 适合相关任务,异构任务应 per-request 服务**。
**✅ 正解验证完成:vLLM per-request 多 LoRA 服务**(一实例服务共享 base + 多专家,每任务路由到自己专家):
| 任务 | 单 adapter | weight-merge d0.5 | **per-request serve** |
|---|---|---|---|
| DC/beer | 0.993 | 0.856 ✗ | **0.993 ✓** |
| RE/RE | 0.772 | 0.774 | **0.771 ✓** |
| AVE | 0.762 | 0.659 ✗ | **0.743 ✓** |
→ **per-request 服务 = 单 adapter 性能 = 零冲突 by construction**;权重合并异构专家则冲突。vLLM 0.6.6(enable_lora/LoRARequest/max_loras)支持。图:`revision_exp/results/figs/moe_merge_vs_serve.png`。
**回应 R2-2.4 / merge 指令**:MoE 推理用 per-request 多 LoRA 服务(零冲突、共享 base 省显存、plug-and-play);weight-merge 仅适合相关任务专家。
报告存 `docs/MOE_merge_research.json`。

### (旧)LoRA merge 计划
- deep-research 进行中(TIES/DARE/LoRAHub/AdaMerging/PEFT add_weighted_adapter/Arrow/PHATGOOSE + vLLM multi-LoRA)。
- 出报告后实现正确的 conflict-free merge + 验证(merge 前后各任务 F1 不掉=无冲突)。
- vLLM:12.43 deepspeed env = **0.6.6.post1**(支持 per-request LoRA 切换);top-m 组合需离线 merge。

## B — matched dense 多任务 7B（AE-4, R2-2.3）⚠️ MIXED,关键诚实发现

dense 多任务 7B(单模型,无 MoE/router,训练于**相同 MELD-DS 选择数据**并集)vs MELD(eval_result 主指标):

| 任务 | MELD | dense-7B | 差(dense−MELD) |
|---|---|---|---|
| AVE/oa_mine | 0.762 | 0.762 | 0 |
| CTA/WebTable(micro) | 0.954 | 0.937 | −0.017 |
| DC/beer | 0.993 | 0.985 | −0.008 |
| DC/hospital | 0.963 | 0.970 | +0.007 |
| DC/rayyan | 0.839 | 0.848 | +0.009 |
| ER/abt-buy | 0.899 | 0.884 | −0.015 |
| ER/amazon-google | 0.776 | 0.783 | +0.007 |
| ER/semi-text-c | 0.301 | 0.243 | −0.058 |
| ER/semi-text-w | 0.716 | 0.794 | +0.078 |
| ER/walmart-amazon | 0.868 | 0.881 | +0.013 |
| ER/wdc | 0.834 | 0.918 | +0.084 |
| RE/RE(micro) | 0.772 | 0.780 | +0.008 |
| (DI amazon/walmart 补跑中;SM/CMS eval handler=0 需修) | | | |

**统计**:dense 胜 7、MELD 胜 4、平 1;**dense 平均略高 ~+0.01**。

**⚠️ 诚实结论(需你定框架)**:matched dense 多任务 7B 在精度上**与 MELD 的 MoE 相当、平均甚至略高**。即:在相同 MELD-DS 数据下,**MoE 结构相对单密集模型没有精度优势**。
- **可辩护框架**:(1) 这恰恰**验证了 MELD-DS 数据选择**(期刊主增量)——无论用 MoE 还是单密集模型,MELD-DS 选的数据都给出强结果;(2) MoE 的价值是**运营性**:plug-and-play 加专家(dense 需全量重训)、独立 refine、router 扩展性(已证 18 experts 稳定),而非精度;(3) 差异 <0.02(除 wdc/semi-text-w),单次无 seed。
- **风险**:若审稿人坚持要 MoE 精度优势,此结果不支持。建议把 journal 卖点明确放在 **MELD-DS(选择)**,MoE 作为正交的部署优势。

## 待做(非 MoE,无需 source)
- [ ] Exp A 汇总 3 数据集(CTA/ER 分析中)
- [ ] B. matched dense 多任务 7B(AE-4, R2-2.3)
- [ ] C. noisy augmentation(AE-6, R1-3)
- [ ] E5. failure case(R1-5,纯 CPU)
- [ ] 写作类草稿(R2-S1/S2/S3、R1-1/1-2/2-2.4 讨论)

## Blocked(需 source / backup survey,先不执行)
- MoE = standalone router + multi-LoRA vLLM + lora merge;router 源码丢失。
- 计划:另起 survey 文档调研已有 MoE 工作 + lora merge 可行性,重搭一套(不动手,等判断)。
- 影响:D(R1-4)、AE-5 经验、R2-2.4 新实验、AE-1 非表格 t-SNE。
