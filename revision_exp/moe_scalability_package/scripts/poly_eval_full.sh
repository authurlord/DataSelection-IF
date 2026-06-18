#!/bin/bash
cd /data/home/wangys/meld_revision/moe_exp
PY=/data/home/wangyaoshu/anaconda3/envs/deepspeed/bin/python
MI=/data/home/wangys/model/Mistral-7B-Instruct-v0.2
OUT=revision_exp/results/expB_probe/probe_f1_full.csv
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for n in "$@"; do
  echo "[full] poly n$n $(date +%H:%M:%S)"
  $PY revision_exp/expMoE_eval_probe.py --mode poly --base $MI --poly poly_probe/n${n}_mhr --tag poly_n${n}_s0 --cap 6000 --out $OUT
done
echo POLY_EVAL_FULL_DONE
