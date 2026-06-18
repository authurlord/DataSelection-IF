# Response Letter [TBD] 回填（实测数据）

> 对应 `docs/MELDDS_TKDE_response_letter_draft.md` 的 `[TBD]` 槽位。✅=已测,⏳=跑中。

## ✅ R2-2.1 / AE-3（聚类开销,Table III）
- WebTable(14,856 样本)聚类(batch-division)wall-clock ≈ **23.7s**,占 **570.9s** 选择时间的 **4.1%**。
- 其他:SimTab 1.8%、ER amazon-google 2.9%、RE 3.5%。聚类全程 ≤4.3%,梯度计算占 75–85%。
- 文案:"On the 14,856-sample WebTable pool, clustering completes in 23.7 s, 4.1% of the 570.9 s selection time, and is therefore not a bottleneck. The 3–10% runtime claim is preserved after accounting for clustering."

## ✅ R2-2.2 / AE-2（Theorem 6 σ 敏感性,Fig X）
三数据集一致(IF method = proposed/DataInf;误差 = batch-IF 与逐样本 IF 的 z-score 平均绝对差):
- **error 随 σ 单调上升**(固定 |B|=8,打乱比例 p 抬 σ):
  - RE: σ_norm 0.850→0.934 ⇒ err 0.590→0.818
  - ER: σ_norm 0.85?→0.934 ⇒ err **0.225→0.771**
  - CTA: σ_norm 0.858→0.932 ⇒ err 0.468→0.871
- **σ(|B|) 随 |B| 单调上升**(语义聚类):RE 0.80→0.91;ER 0.56→0.82;CTA 0.82→0.92。
- **random vs 语义聚类(|B|=8)误差**:RE 0.84/0.59;ER **0.80/0.23**;CTA 0.875/0.468。
- 文案:"Random clustering raises within-batch σ and the batch-IF approximation error by up to 3.5× relative to semantic k-center clustering at matched |B|=8 (ER: error 0.80 vs 0.23). The error grows monotonically with σ, and σ itself grows with |B|, so the approximation does not improve as 1/√|B| under a constant σ; instead σ(|B|)/√|B| yields the sweet-spot consistent with Table V."

## ✅ R2-S4（Table III 初始化时间）
- DataInf 1200s/22GB = full-7B 逐样本梯度计算+存储;MELD-DS 115s/3.4GB = 0.5B proxy warm-up + 批级梯度。差异源于 proxy 规模 + 批级,而非省略 IF 步骤。

## ✅ R1-5（failure case,附录小节）
- ER/semi-text-c F1=0.301:类不平衡(正例13.9%)+ 高假阳(FP689/FN354),难半结构化匹配过度判 match。
- AVE/oa_mine exact=0.763:抽取值不精确(格式/释义)。
- CTA/SimTab:常见类 acc 0.808 vs 稀有类 0.551 → 拉低 macro-F1(0.734)。
- 文案:MELD 强在闭域/结构受限任务,弱在极端类不平衡半结构化匹配、开放域细粒度抽取、长尾稀有类型。

## ✅ AE-1 / R1-1（ITS,Fig 2,表格部分）
- 16 个异构 DP 任务的 LoRA task vector(各 688,128 维)内在维度 **8(90%方差)/11(95%)**;PC1=34.6%, PC1-6=85.4%。
- 文案:"Despite a 688k-dimensional LoRA parameter space, the per-task adaptation vectors of 16 heterogeneous DP tasks are confined to an 8-dimensional subspace (90% variance), empirically instantiating the intrinsic-task-subspace property in the tabular-dense DP regime."
- ⏳ 非表格点(math/grammar/events-classification)待加入 t-SNE。

## ⏳ R2-2.3 / AE-4（dense 多任务 7B,Table I）
- 训练中(12.43,Mistral-7B LoRA,31166 样本并集,3 epochs)。完后逐任务推理+评测,对比 MELD。

## ⏳ R1-3 / AE-6（noisy augmentation,Fig Y）
- ER/amazon-google {0,10,20,30}% 噪声 expert 训练中(51.11);CTA 待跑。完后出 F1-噪声降解曲线。

## ⏳ R1-4 / AE-5 / R2-2.4（MoE/router/scalability/lora-merge）
- B 训练后在 12.43 实跑(用户授权)。ITS 非表格 + task-count + routing + lora-merge。
