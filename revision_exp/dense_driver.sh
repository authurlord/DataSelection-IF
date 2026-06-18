#!/bin/bash
# Dense multi-task train+eval driver for 51.11. Pinned to ONE gpu. Usage: dense_driver.sh <GPU> <n...>
# Trains pre-generated per-n yamls (revision_exp/scaling_cfg/_dense_n${n}_s0.yaml).
set -e
GPU=$1; shift; NS="$@"
MI=/data/yanmengyi/huggingface/Mistral-7B-Instruct-v0.2
cd /data/yanmengyi/DataSelection-IF
source /data/yanmengyi/anaconda3/etc/profile.d/conda.sh; conda activate deepspeed
for n in $NS; do
  OUT=/data/yanmengyi/DataSelection-IF/revision_exp/lora/dense-probe/n${n}_s0_1ep
  YAML=revision_exp/scaling_cfg/_dense_n${n}_s0.yaml
  echo "[dense] train n=$n gpu=$GPU $(date +%H:%M:%S) -> $OUT"
  CUDA_VISIBLE_DEVICES=$GPU llamafactory-cli train $YAML
  echo "[dense] eval n=$n $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU python revision_exp/expMoE_eval_probe.py --mode dense --base $MI \
      --adapter $OUT --tag dense_n${n}_s0 --cap 400 --out revision_exp/results/expB_probe/probe_f1.csv
done
echo DENSE_DRIVER_DONE_$GPU
