#!/usr/bin/env bash

ABLATION=$1
MODEL=$2
DEVICE=$3

max_jobs=$4

num_jobs="\j"

case $ABLATION in
margin_type)
  for memory in 1000
  do
    for margin_w in 0.25
    do
      for margin_type in 'mean' 'max_mean'
      do
        for margin in 1 0.5 0.25 0.1
        do
        while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
            wait -n
          done
          python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=margin device=$DEVICE method.mem_size=$memory method.past_task_reg=$margin_w method.gamma=1 hydra=search +wadnb_tags=[margin_type_ablation] method.margin_type=$margin_type method.margin=$margin head=margin_head &
        done
      done
    done
  done
;;
scaler)
  python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=margin device=$DEVICE method.mem_size=1000 method.past_task_reg=0.25 method.gamma=1 hydra=search +wadnb_tags=[margin_scale_ablation] head=margin_head +head.scale=False
;;
future)
  for future_classes in 10 20 30 50
  do
    while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
      wait -n
    done
    python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=margin device=$DEVICE method.mem_size=1000 method.past_task_reg=0.25 method.gamma=1 hydra=search +wadnb_tags=[margin_future_ablation] head=margin_head +head.future_classes=$future_classes &
  done
;;
logit)
  for margin in 0.5 0.75 1
  do
    while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
      wait -n
    done
    python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=margin device=$DEVICE method.mem_size=1000 method.past_task_reg=0.25 +model.regularize_logits=True method.margin_type=fixed method.margin=$margin method.gamma=1 hydra=search +wadnb_tags=[margin_logits_ablation] head=margin_head &
  done
;;
sigmoid)
  for a in 0 1 2.5 5 10 20
  do
    for b in 1 0.5 1 2.5 5
    do
      while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
        wait -n
      done
      python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=margin device=$DEVICE method.mem_size=1000 method.past_task_reg=0.25 +model.regularize_logits=True method.gamma=1 hydra=search +wadnb_tags=[margin_simgoid_ablation] head=margin_head head.a=$a head.b=$b experiment=base2 &
    done
  done
;;
tradeoff)
for memory in 200 500 1000 2000 4000 6000 10000
do
  for past_margin_w in 0.5 0.25 0.1 0.05 0.025 0.01
  do
    while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
      wait -n
    done
    python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=margin_cifar10 device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=1 hydra=search +wadnb_tags=[sp_to_ablation] method.margin_type=adaptive  experiment=base1 head=margin_head &
  done
done
;;
*)
  echo -n "Unrecognized ablation experiment"
esac
