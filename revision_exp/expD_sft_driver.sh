#!/bin/bash
# expD_sft_driver.sh <GPU> <pcts...>  : per p: gen yaml, LoRA SFT 1ep on sel_pN, eval ER F1

GPU=$1; shift; PCTS="$@"
cd /data/yanmengyi/DataSelection-IF
source /data/yanmengyi/anaconda3/etc/profile.d/conda.sh; conda activate deepspeed
MI=/data/yanmengyi/huggingface/Mistral-7B-Instruct-v0.2
for pct in $PCTS; do
  OUT=/data/yanmengyi/DataSelection-IF/revision_exp/seed_study/expD_lora/p${pct}
  YAML=revision_exp/scaling_cfg/_er_p${pct}.yaml
  sed -e "s#__DATASET__#expD_sel_p${pct}#" -e "s#__OUT__#${OUT}#" revision_exp/scaling_cfg/er_sft_template.yaml > $YAML
  echo "[expD] train p=$pct gpu=$GPU $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU llamafactory-cli train $YAML
  echo "[expD] eval p=$pct $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU python revision_exp/expD_eval_er.py --base $MI --adapter $OUT \
      --tag p${pct} --out revision_exp/results/expD_cohesion_f1_raw.csv
done
echo EXPD_DRIVER_DONE_$GPU
