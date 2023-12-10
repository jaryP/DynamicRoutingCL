#!/usr/bin/env bash
METHOD=$1
MODEL=$2
DEVICE=$3

max_jobs=$4
num_jobs="\j"

case $METHOD in
der)
  for memory in 200 500 1000 2000
    do
    for alpha in 0.1 0.2 0.5 0.8 1.0
    do
      for beta in 0.1 0.2 0.5 0.8 1.0
      do
        python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der device=$DEVICE head=linear method.alpha=$alpha method.beta=$beta experiment=dev method.mem_size=$memory hydra=search +wadnb_tags=[grid_search]
      done
    done
  done
;;
ewc)
  for lambda in 1 10 100 1000
  do
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ewc_default optimizer=adam  device="$DEVICE" method.ewc_lambda=$lambda experiment=dev hydra=search
  done
;;
oewc)
  for lambda in 1 10 100 1000
  do
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=oewc_default optimizer=adam  device="$DEVICE" method.ewc_lambda=$lambda experiment=dev hydra=search
  done
;;
#routing)
##  routing_3l_convblock
##  routing_3l_convblock_invusage
#  for memory in 200 500 1000 2000
#  do
#    for margin in 0.1 0.2 0.3 0.5
#    do
#      for past_margin_w in 1 0.5 0.1 0.01 0.001
#      do
##        for future_margin in 1 2 3 4 5
#        for future_margin in 3
#        do
##            for gamma in 1 0.5 0.1
#            for gamma in 1 0.5 0.1 0.0
#            do
#              python main.py +scenario=cil_cifar10_5 +model=$MODEL +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=dev method.mem_size=$memory method.margin=$margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=0.5 method.gamma=$gamma hydra=search
#            done
#          done
#        done
#      done
#    done
#;;
margin)
  for memory in 200 500 1000 2000
  do
    for past_margin_w in 0.5 0.1 0.01 0.05
    do
          for gamma in 0.5 1 2
          do
          for alpha in 0
          do
            python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=$gamma +method.alpha=$alpha hydra=search +wadnb_tags=[grid_search] experiment=dev head=margin_head wandb_prefix=lenovo_
          done
          done
      done
    done
;;
logit_d)
  for memory in 200 500 1000 2000
  do
    for alpha in 0.1 0.25 0.5 0.75 1
    do
      while (( ${num_jobs@P} >= ${max_jobs:-1} )); do
        wait -n
      done
      python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=logit_d device=$DEVICE method.mem_size=$memory method.alpha=$alpha hydra=search experiment=dev head=incremental +wadnb_tags=[grid_search] &
    done
  done
;;
*)
  echo -n "Unrecognized method"
esac
