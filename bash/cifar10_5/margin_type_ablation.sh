#!/usr/bin/env bash

MODEL=$1
DEVICE=$2

max_jobs=$3

num_jobs="\j"

for memory in 500
do
  for past_margin_w in 0.25
  do
    for margin_type in 'mean' 'max_mean'
    do
      for past_margin in 1 0.5 0.25 0.1
      do
        while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
          wait -n
        done
        python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin_cifar10 device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=1 +method.alpha=0 hydra=search +wadnb_tags=[margin_type_ablation] method.margin_type=$margin_type method.past_margin=$past_margin  +method.rehearsal_metric='kl' experiment=base1 head=margin_head &
      done
    done
  done
done
