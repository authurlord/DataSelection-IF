# 资源使用限制 & 三服务器环境现状（MELD-DS Revision）

> 落盘时间：2026-06-15。供后续所有实验遵守。

## 硬性资源限制（用户规定）

| 服务器 | 地址 / 账户 | GPU 使用许可 | 说明 |
|---|---|---|---|
| **51.10**（本地 `jcyj02`） | 本机 | **禁止使用 GPU**（除非必要） | 只用于写代码、版本控制、scp 中转。RTX 3090 共享且繁忙 |
| **51.11**（`distr-node2`） | yanmengyi@192.168.51.11 | **全部资源可用** | **主实验机** |
| **12.43**（`12-43`） | wangys@192.168.12.43 | **仅 cuda 6,7，且不得抢占他人** | 备用；需 bf16/FA2 时用 |

口令：51.11 = `Sics32375@0110&`（已配置免密 key，BatchMode 可直连；12.43 亦免密可连）。

## 51.11 主实验机硬件/软件现状

- **GPU**：4 × Tesla **V100-PCIE-32GB**，compute capability **7.0** → **不支持 bf16**、**不支持 flash-attn2**（FA2 需 Ampere cc≥8.0）。当前 4 卡全空闲。
  - 训练/推理一律用 **fp16**（yaml 里 `bf16:true` 须改 `fp16:true`）。
  - vllm 推理须加 **`dtype=half`**（否则 cc7.0 报 bf16 错；这不是 glibc 问题，glibc 2.17 + vllm 0.5.4 可正常 import/运行）。
  - 无 flash_attn / liger_kernel（训练 yaml 的 `enable_liger_kernel:true`、`neat_packing` 在 V100 上须关闭或忽略）。
- **RAM**：471GB，可用 ~463GB → **OOM 风险极低**（仍需在大梯度聚合时监控）。
- **conda 环境 `deepspeed`**（`/data/yanmengyi/anaconda3/envs/deepspeed`）：
  - Python 3.11、llamafactory **0.9.2 editable**（在 `/data/yanmengyi/LLaMA-Factory/src`，**已含直接读 json 的定制改动**）、torch 2.4.0+cu121、transformers 4.47.1、peft 0.12.0、deepspeed 0.16.2、vllm 0.5.4。
  - 激活：`source /data/yanmengyi/anaconda3/etc/profile.d/conda.sh && conda activate deepspeed`
- **模型路径（51.11）**：
  - Mistral-7B：`/data/yanmengyi/huggingface/Mistral-7B-Instruct-v0.2`
  - Qwen2.5-0.5B proxy：`/data/yanmengyi/models/Qwen2.5-0.5B-Instruct`
  - bge-large-en-1.5：`/data/yanmengyi/sentence_transformer_model/bge-large-en-1.5`
  - ⚠️ config 里硬编码的 `/data/home/wangys/...`、`/public/...`、`/home/yanmy/...` 全部 MISSING，须重映射到上面路径。
- **repo**：`/data/yanmengyi/DataSelection-IF`（git HEAD=`4499a20`，**有未提交的活改动**，以工作树为准）。

## 12.43 现状（备用）

- 8 × 80GB（cc≥8.0，A100/A800 级，支持 bf16+FA2）。GPU0/1 被他人占 ~65GB，**仅用 cuda 6,7**。
- base 模型在 `/data/home/wangys/model/{Mistral-7B-Instruct-v0.2, Qwen2.5-0.5B-Instruct}`（即 ckpt adapter_config 里硬编码的原始路径）。
- `/data` 21T、空闲 7.2T。
- `/data/home/wangys/out/DataSelection-IF` 只是一个输出 ckpt 目录，**不是代码 repo**；无 grad/Influence/selection 遗产。

## 遗产中间数据搜索结论（三台都搜过）

- ❌ **原始梯度 `grad/`、`grad_single/`：三台均无**（算完即删的临时产物，必须重算；0.5B proxy 很便宜）。
- ✅ **51.11 保留**：`Influence/*/{batch.pkl, batch_single.pkl, result.pkl, score.pkl}`（全数据集的 k-center 分组、逐样本分组、batch-IF）、`selection/*/{FL-Score, Total-Score, Cluster-Rank}.pkl`（全数据集）。→ 作为**复算的验证基准**。
- ✅ 模型、数据 pool（train/*/*-train.json）、init adapter、120 个 LoRA 权重均在。

## 工作约定

- 代码在 51.10 本地 `reproduce` 分支写（版本控制），scp 到 51.11 跑；51.11 上新产物写到 `revision_exp/` 子目录，**不覆盖原有 train-select / output / Influence**。
- 监控：每次大 GPU/内存任务前后 `nvidia-smi` + `free -g`。
- 不提交 GitHub（用户要求）。
