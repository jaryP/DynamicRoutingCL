#!/usr/bin/env bash

ABLATION=$1
MODEL=$2
DEVICE=$3

max_jobs=$3

num_jobs="\j"

case $ABLATION in
margin_type)
  for memory in 500
  do
    for margin_w in 0.25
    do
      for margin_type in 'mean' 'max_mean'
      do
        for margin in 1 0.5 0.25 0.1
        do
          while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
            wait -n
          done
          python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin_cifar10 device=$DEVICE method.mem_size=$memory method.past_task_reg=$margin_w method.gamma=1 +method.alpha=0 hydra=search +wadnb_tags=[margin_type_ablation] method.margin_type=$margin_type method.margin=$margin head=margin_head &
        done
      done
    done
  done
;;
scaler)
  python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin_cifar10 device=$DEVICE method.mem_size=500 method.past_task_reg=0.25 method.gamma=1 hydra=search +wadnb_tags=[margin_scale_ablation] head=margin_head head.scale=False
;;
future)
  for future_classes in 10 20 30 50
  do
    python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin_cifar10 device=$DEVICE method.mem_size=500 method.past_task_reg=0.25 method.gamma=1 hydra=search +wadnb_tags=[margin_future_ablation] head=margin_head head.future_classes=$future_classes
  done
;;
logit)
  for margin in 0.5 0.75 1
  do
    while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
      wait -n
    done
    python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin_cifar10 device=$DEVICE method.mem_size=500 method.past_task_reg=0.25 model.regularize_logits=True method.margin_type=fixed margin_type.margin=$margin method.gamma=1 hydra=search +wadnb_tags=[margin_logits_ablation] head=margin_head &
  done
;;
*)
  echo -n "Unrecognized ablation experiment"
esac