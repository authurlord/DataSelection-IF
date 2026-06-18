#!/bin/bash
# Poly+MHR train+eval driver for 12.43. Pinned to ONE gpu (cuda6 or 7). Usage: poly_driver.sh <GPU> <n...>
set -e
GPU=$1; shift; NS="$@"
D=/data/home/wangys/meld_revision/moe_exp
PY=/data/home/wangyaoshu/anaconda3/envs/deepspeed/bin/python
MI=/data/home/wangys/model/Mistral-7B-Instruct-v0.2
cd $D
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for n in $NS; do
  OUT=poly_probe/n${n}_mhr
  echo "[poly] train n=$n gpu=$GPU $(date +%H:%M:%S)"
  $PY revision_exp/poly_train.py --base $MI --data_glob "revision_exp/poly_data/n${n}/task-*.json" \
      --out $OUT --n_skills 8 --n_splits 4 --epochs 1 --enable_thinking 0 --bsz 2 --cutoff 1024
  echo "[poly] eval n=$n $(date +%H:%M:%S)"
  $PY revision_exp/expMoE_eval_probe.py --mode poly --base $MI --poly $OUT \
      --tag poly_n${n}_s0 --cap 400 --out revision_exp/results/expB_probe/probe_f1.csv
done
echo POLY_DRIVER_DONE_$GPU
