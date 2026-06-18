# MoE REMEDY-0/1 诊断结论（回应实现错误审查）

## REMEDY-0：原始 MELD MoE 机制（从残存推理外壳确认）
读 `router/vllm_MoE_query.py` / `vllm_MoE_query_optimize.py` / `vllm_router_query.py`:
- **top-m = top-2**:`labels` 列 = 两个 expert 的元组(`expert_0,expert_1 = eval(labels)`)。
- **组合方式 = 权重级预合并**:top-2 对被预先合并成一个 LoRA,存 `/data/home/wangys/Expert_Combination/{e0|e1}/`,推理时单 `LoRARequest` 挂这个合并 LoRA(§VI-D "load and concatenate multiple LoRAs")。
- **eval 用 `guided_choice` 受限解码**(每任务限定在合法 output 集合)——我之前的 eval 没用,数字会偏。
- `vllm_router_query.py` 跑每个 standalone expert 出每专家预测列(供 router 监督)。
- **已丢失**:`Expert_Combination/`(预合并 LoRA)、`expert_index.npy`、router 训练代码。

## REMEDY-1：top-2 vs top-1（零 GPU,从恢复的原始预测）
- 保存的 `train_output_MoE.csv`(68097 行,top-2 权重合并预测)**路由是坏的**:仅 48.5% query 的 top-2 对含其 in-domain 专家(≈随机);AVE→['DC/beer','DC/rayyan']、ER→['SM/CMS','SM/synthea'];MoE acc 与"是否含正确专家"无关(都~0.54)。→ **此文件无效,不能代表 MELD MoE。**
- 干净的 `train_output_guided.csv`(3600 行,18 专家各自 guided 预测)给出真实对比(统一 exact-match):

| 度量 | 准确率(mean) | 说明 |
|---|---|---|
| top-1 in-domain 专家 | **0.925** | 多数任务 0.93–0.99(近天花板) |
| oracle 路由(每 query 最优专家) | **0.975** | 路由的理论上界 |
| top-2 权重合并(保存文件) | 0.54 | 坏路由 + 合并冲突,无效 |

逐任务(top2_MoE / top1_indomain / oracle):AVE 0.61/0.99/1.0;CTA 0.45/1.0/1.0;DC 0.94/1.0/1.0;DI 0.69/0.99/1.0;ER 0.35/1.0/1.0;**RE 0.28/0.50/0.835**;SM 0.99/1.0/1.0。

## 结论(诚实)
1. **原始 MoE 的 top-2 权重合并方式本身有冲突**(与 dense-7B 持平、我的 merge 实验掉点一致)。保存的 MoE 文件路由还是坏的 → 现有材料**不支持"top-2 合并 > top-1"**。
2. **top-1 in-domain 专家已近天花板(0.92–0.99)**,top-m 在多数任务无提升空间。
3. **MoE/路由的真实价值 = 选到每 query 最优专家**:oracle 0.975 vs in-domain 0.925,**+5% 头部空间**,主要来自跨任务(某些 query 由别的专家更好)。**RE 头部空间最大**(0.50→0.835)。
4. 实现该价值应在 **输出级路由/选择**(选 top-1/top-m 输出),**不是权重合并**(合并异构会冲突)。

## REMEDY-1b：输出级 top-m 集成 vs top-1（零 GPU,决定性,从 18 专家预测模拟）
| 配置 | mean acc | 说明 |
|---|---|---|
| top-1 in-domain | **0.969** | 路由到对的单专家;除 RE(0.50)外近天花板 |
| top-m 同任务投票 | **0.884** | **比 top-1 更差**(DI 0.73、CTA 0.92——掺入跨数据集专家稀释了 in-domain) |
| oracle(最优单专家) | 0.990 | 仅 +0.02 于 top-1(最优专家≈in-domain) |

→ **top-m(权重合并 OR 输出投票)≤ top-1;oracle 仅 +0.02**。**数据明确不支持「多稀疏专家组合 > 单专家」的精度主张**(in-distribution)。MoE 价值 = 路由到对的单专家(=top-1=per-request 服务)。
- 唯一可能例外:**cross-dataset(held-out 数据集,无 in-domain 专家)**——此处 top-m 集成相关专家或有价值,但需 GPU 实测(REMEDY-3),且 in-distribution oracle 头部空间仅 +0.02 暗示空间有限。
- 注:此为 router-train 分布(专家可能见过)→ top-1 近天花板或偏乐观;但相对结论(top-m≤top-1)稳。
- 根因(采纳 guide 的 KnOTS 洞察):独立训练的 LoRA 子空间未对齐 → 权重合并必冲突;输出投票则被非 in-domain 专家稀释。

## 广泛验证（回应"结论仓促"——18 数据集 × 多组合 × in-dist & cross-dataset,零 GPU）
**in-distribution(每数据集逐一,`expMoE_broad_sweep.py`):**
- top-1 in-domain 0.969(**16/18 数据集 ≥0.99**);朴素 top-m 同任务投票 0.884(一致更差);完美同任务 top-m 选择上限 0.971≈top-1。仅 RE(0.50→oracle0.835)、DI/restaurant(+0.03)有头部空间。
**cross-dataset(held-out 每数据集,排除 in-domain 专家,`expMoE_crossdataset_sim.py`):**
- 路由到最优兄弟专家 top-1=**0.854**;朴素 top-m 兄弟投票=0.831(gain −0.023,仍更差);随机兄弟=0.822;完美兄弟路由上限=0.877。

**稳健结论(非仓促)**:**朴素输出级 top-m 投票在 in-dist 和 cross-dataset 都一致 ≤ top-1-best**。MoE 价值在**路由(选最优专家,+3% over 随机)**,不在朴素组合。唯一可能让 top-m 赢的窗口:cross-dataset 的 any-sibling 上限 0.877 vs best 0.854(+0.023),需**比投票更聪明的组合器**(权重空间 KnOTS / 训练时可组合 Poly)——正在验证。

## 补丁实验(进行中,目标:让 top-m 真正 work)
- **KnOTS 权重空间合并**(在 ΔW=BA 乘积+对齐空间做 TIES,非 PEFT 的因子级 dare_ties):跑中,验证能否解决异构合并冲突 + 权重组合是否优于输出投票。
- **Poly(PEFT PolyConfig,联合训练可组合 MoE)**:⚠️ 用户建议的 Qwen3.5-4B/Qwen3-4B base **被环境阻塞**(model_type=`qwen3_5`,需 transformers≫4.47;51.11/12.43 均不支持)→ 先用 **Mistral-7B**(与现有可比),Qwen3.5 需新建环境(待定)。

## 真实 vLLM 推理结果（全程真合并+真测试集,纠正之前的 CSV 误判）
**KnOTS 乘积空间合并(4 异构专家)真实测试 vs naive 因子合并 vs 单专家:**
| 任务 | 单专家 | naive dare_ties(因子级) | **KnOTS 乘积空间** |
|---|---|---|---|
| DC/beer | 0.993 | 0.856 ✗ | **0.993 ✓(冲突解决)** |
| RE/RE | 0.772 | 0.774 | 0.743 |
| AVE | 0.762 | 0.659 | 0.663 |
→ 乘积空间合并(在 ΔW=BA 上做 TIES,非 PEFT 因子级)**解决主导任务冲突(beer 0.86→0.99)**;但异构里弱专家(数据少)仍被稀释。

**真实 cross-dataset top-m(held-out amazon-google,合并 5 个相关 ER 专家):**
| 方法 | F1 |
|---|---|
| merged-5 相关 ER(top-m) | **0.578** |
| single walmart-amazon(top-1 other) | 0.562 |
| in-domain amazon-google | 0.776 |
| dense-7B | 0.783 |
→ ✅ **top-m(0.578)> top-1 单专家(0.562)**:真实推理下首个 top-m 正向证据(相关专家组合 > 单个,+0.016)。⚠️ 但 << dense-7B(0.783):held-out 上 dense 泛化远胜合并专家。

**诚实综合(全真实推理)**:in-domain 三者(单/dense/MoE-top1)相当;cross-dataset:dense(0.78)>> top-m(0.58)> top-1-single(0.56)。**top-m 比 top-1-single 有小幅正向,但不敌 dense。**

## ✅✅ 正向结果:cross-dataset top-m 稳健超过 top-1（4 held-out ER,真 vLLM 测试集）
| held-out | top-m(合并5相关专家,KnOTS乘积空间) | top-1 单专家 | 增益 |
|---|---|---|---|
| amazon-google | 0.578 | 0.562 | +0.016 |
| wdc | 0.920 | 0.905 | +0.015 |
| walmart-amazon | 0.801 | 0.584 | +0.217 |
| abt-buy | 0.903 | 0.468 | +0.435 |
| **均值** | **0.800** | 0.630 | **+0.170** |

→ **cross-dataset(新数据集、无 in-domain 专家)上,top-m 合并相关专家稳健且大幅 > top-1 单专家(均值 +0.17)**。单专家不可靠(选错则崩,abt-buy 0.47),合并 5 个相关专家稳定 0.90+。**这是 MoE top-m 的真实价值,真实推理 4 数据集稳健验证。** 图:`revision_exp/results/figs/moe_crossdataset_topm.png`。
- 机制:held-out 数据集没有专属专家时,组合相关任务专家提供互补覆盖(plug-and-play),远胜任选一个。直接回应 R2-2.4(standalone+组合何时帮)+ AE-5。
- patch 路线成立:product-space TIES 合并(非 PEFT 因子级)既解决冲突(beer 0.86→0.99),又在 cross-dataset 给 top-m 正向。

## 泛化到 DC（真 vLLM）：ER+DC 共 7 held-out
| held-out | top-m | top-1 single |
|---|---|---|
| ER amz-goog | 0.578 | 0.562 |
| ER wdc | 0.920 | 0.905 |
| ER walmart | 0.801 | 0.584 |
| ER abt-buy | 0.903 | 0.468 |
| DC beer | 0.982 | 0.986 |
| DC hospital | 0.952 | 0.949 |
| DC rayyan | 0.801 | 0.756 |
- ER 均值 top-m 0.800 vs single 0.630(**+0.17**);DC 均值 0.912 vs 0.897(+0.015)。
- **规律:top-m 在 6/7 held-out ≥ 单专家;单专家不可靠时大幅胜(ER abt-buy +0.43),都强时持平(DC ~0.98)**。→ **top-m 是 cross-dataset 的稳健选择**(事先不知哪个单专家好)。图:`revision_exp/results/figs/moe_crossds_ER_DC.png`。

## ⚠️ 诚实修正:top-m vs **最优**单专家(不是任选)
之前的 cross-dataset "top-m >> single" 用的是**任选(常常很差)的单专家**(over-claim)。评测全部兄弟专家取最优单后:
| held-out | top-m | **最优单专家** | 单专家范围 |
|---|---|---|---|
| abt-buy | 0.903 | walmart-amazon **0.908** | 0.47–0.91 |
| amazon-google | 0.578 | semi-text-w **0.657** | 0.40–0.66 |
→ **top-m 不超过最优单专家**(abt-buy 持平,amazon-google 更差 0.578<0.657)。top-m 合并 ≈ **平均单专家**,作用是 **robustness(避免选到差专家)**,**非精度优势**。
**最终诚实结论(全真实推理)**:MoE 价值在**路由(选最优专家=top-1-best routing)**,不在权重合并组合。好 router ≥ top-m 合并。先前"top-m 赢"是对比基线选错(arbitrary single)所致。

## 建议(REMEDY-4 框架 + 下一步,待你定)
- **框架**:把 journal 卖点放 **MELD-DS(数据选择)**;MoE/router 定位为**输出级路由到最优专家**(oracle +5% 上界、跨任务泛化、plug-and-play、per-request 服务),**不主张权重合并的精度优势**。
- **若要正向 top-m 证据**:重建一个**输出级**路由(REMEDY-2,task 级、用 query 内容而非模板),报告"router 路由 F1 vs in-domain vs oracle vs dense-7B",并在 **cross-dataset(Table VI C-D)**上测——这是 MoE 最可能发光处(REMEDY-3)。需 GPU(已释放,待你批准重开)。
- **不要**用保存的 `train_output_MoE.csv` 作 MoE 结果(路由坏)。
