#!/usr/bin/env bash
METHOD=$1
MODEL=$2
DEVICE=$3

max_jobs=$4
num_jobs="\j"

case $METHOD in
der)
  for memory in 200 500 1000 2000 5000
    do
    for alpha in 0.1 0.2 0.5 0.8 1.0
    do
      for beta in 0.1 0.2 0.5 0.8 1.0
      do
        while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
          wait -n
        done
        python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der device=$DEVICE method.alpha=$alpha method.beta=$beta experiment=dev method.mem_size=$memory hydra=search head=linear +wadnb_tags=[grid_search] head.out_features=200 &
      done
    done
  done
;;
margin)
for memory in 200 500 1000 2000 5000
do
  for past_margin_w in 1 0.5 0.25 0.1 0.05 0.025 0.01
  do
    while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
      wait -n
    done
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=1 +wadnb_tags=[grid_search_margin] method.margin_type=adaptive experiment=dev hydra=search &
    sleep 1
  done
done
;;
lode)
  for memory in 200 500 1000 2000 5000
    do
    for rho in 0.1 0.2 0.5
    do
      python main.py +scenario=cil_tyn_10 model="$MODEL" +training=cifar10_5 +method=lode device=$DEVICE head=incremental method.rho=$rho experiment=dev method.mem_size=$memory hydra=search +wadnb_tags=[grid_search]
    done
  done
;;
logit_d)
  for memory in 200 500 1000 2000 5000
  do
    for alpha in 0.1 0.25 0.5 0.75 1
    do
      while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
        wait -n
      done
      python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=logit_d device=$DEVICE method.mem_size=$memory method.alpha=$alpha hydra=search experiment=dev head=incremental +wadnb_tags=[grid_search] &
    done
  done
;;
*)
  echo -n "Unrecognized method"
esac
