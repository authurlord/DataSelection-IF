#!/bin/bash
cd /data/home/wangys/meld_revision/moe_exp
PY=/data/home/wangyaoshu/anaconda3/envs/deepspeed/bin/python
MI=/data/home/wangys/model/Mistral-7B-Instruct-v0.2
OUT=revision_exp/results/expB_probe/probe_f1_full.csv
GPU=$1; shift; NS="$@"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for n in $NS; do
  echo "[poly-s1] train n=$n $(date +%H:%M:%S)"
  $PY revision_exp/poly_train.py --base $MI --data_glob "revision_exp/poly_data/n${n}/task-*.json" --out poly_probe/n${n}_mhr_s1 --n_skills 8 --n_splits 4 --epochs 1 --enable_thinking 0 --bsz 2 --cutoff 1024 --seed 1
  echo "[poly-s1] eval n=$n $(date +%H:%M:%S)"
  $PY revision_exp/expMoE_eval_probe.py --mode poly --base $MI --poly poly_probe/n${n}_mhr_s1 --tag poly_n${n}_s1 --cap 6000 --out $OUT
done
echo POLY_S1_DONE_$GPU
