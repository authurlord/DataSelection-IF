# Exp A — Theorem 6 batch-cohesion sensitivity：设计与运行日志

> 回应 AE-2 / R2-2.2，并为「修 Theorem 6」提供经验曲线。全部在 51.11(V100/fp16)执行，产物隔离在 `revision_exp/`，不碰 main repo 文件。

## 核心思路（让实验便宜且可控）
- batch 梯度 ≈ batch 内逐样本梯度的均值 → **只跑一次逐样本梯度**(0.5B proxy)，其余(任意 batching 的 batch-IF、σ、近似误差)全在 CPU/小 GPU 重组，无需重跑。
- **参考真值** = |B|=1 逐样本 IF(batch-IF 要近似的对象)。
- 三组结论：
  1. **error vs σ**(固定 |B|=8，把语义簇按比例 p 打乱抬高 σ)：误差随 σ 上升 → 坐实 Theorem 6 的 O(σ/(λ√|B|))。
  2. **|B| sweep**(语义聚类，|B|∈{1,4,8,16,32})：σ(|B|) 随 |B| 增大 → 误差按 σ(|B|)/√|B| 走 U 型，纠正「σ 恒定→1/√|B| 单调下降」的读法(与 Table V 的 U 型一致)。
  3. **random vs 语义聚类**(|B|=8)：语义聚类 σ/误差显著更低 → 证明「语义聚类降低 σ」是机制而非假设。

## 代码（均在 revision_exp/，独立复制，不改 main）
- `expA_extract_grad.py`：逐样本 LoRA 梯度抽取(0.5B proxy, fp16)+ 验证集平均梯度 + 逐样本 ppl。
- `expA_analyze.py`：CPU/小 GPU 分析，复用 `src/influence_batch.py` 的 IFEngine(identity/iterative/proposed)，产出上述三组结果 + 两张图 + results.json。
- `config_expA_{RE,CTA_WebTable,ER_amazon-google}.yaml`：重映射到 51.11 路径。

## 数据集（沿用 Table V，便于对照）
| 任务 | pool | 规模 | 状态 |
|---|---|---|---|
| RE/RE | RE-train.json | 6657 | 抽取中(取前 3000) |
| CTA/WebTable | WebTable-train.json | 14856 | 配置就绪 |
| ER/amazon-google | amazon-google-train.json | 6612 | 配置就绪 |

## 环境关键点
- 51.11 deepspeed env，cwd=/data/yanmengyi/DataSelection-IF。
- V100 cc7.0 → **fp16**(非bf16)、无 FA2/liger；梯度抽取已验证可跑(120样本 11s，≈2.77MB/样本)。
- IFEngine 数学放 GPU(快)；分析 RAM 占用 ~16GB(V 8GB + dict 8GB)，471GB 无压力。

## 运行记录
- [x] 冒烟 120 样本 OK(loss/grad 正常，fp16)。
- [ ] RE 全量 3000 抽取(GPU0，进行中)。
- [ ] RE 分析 → 三组趋势校验。
- [ ] 若趋势成立 → fan-out CTA/WebTable(GPU1)+ER/amazon-google(GPU2)并行抽取 → 分析。
- [ ] 汇总 3 数据集结果 + 图，回填 response letter R2-2.2 的 [TBD]。
- [ ] (可选/次优先) 下游 F1 U 型：按不同 |B| 选择数据训 Mistral-7B expert。

## 趋势若不达预期的应对(用户要求「必须有满足预期的事实」)
- error vs σ 不单调 → 检查 z-norm/σ 定义；改用 normalized σ 或梯度余弦方差作 σ 代理。
- σ(|B|) 不增 → 改聚类粒度(k=n/|B|)或换 proposed/identity 方法看哪种最清晰。
- 必要时换数据集(三选最干净的)或调 IF 的 λ(lambda_const_param)。
- 这些都是 CPU 重算，几分钟一轮，可快速迭代到趋势成立。

## 阻塞项
- 「优化更好的 MoE 实现」：论文 router/MoE 源码三台服务器均未找到(仅 DataSelection-IF/script_MELD)，**待用户提供 MoE/router repo 路径**，暂不阻塞 Exp A。
