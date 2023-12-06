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
        while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
          wait -n
        done
        python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=der_default device=$DEVICE method.alpha=$alpha method.beta=$beta experiment=dev method.mem_size=$memory hydra=search head=linear +wadnb_tags=[grid_search] &
      done
    done
  done
;;
margin)
for memory in 200 500 1000 2000
do
  for past_margin_w in 1 0.5 0.25 0.1 0.05 0.025 0.01
  do
    while (( ${num_jobs@P} >= $((max_jobs + 0)) )); do
      wait -n
    done
    python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=margin_cifar10 head=margin_head device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=1 +method.alpha=0 +wadnb_tags=[grid_search_margin] method.margin_type=adaptive +method.rehearsal_metric='kl' experiment=dev hydra=search &
  done
done
;;
#ewc)
#  for lambda in 1 10 100 1000
#  do
#    python main.py +scenario=cil_cifar100_10 +model="$MODEL" +training=cifar10_5 +method=ewc_default optimizer=adam  device="$DEVICE" method.ewc_lambda=$lambda experiment=dev hydra=search
#  done
#;;
#oewc)
#  for lambda in 1 10 100 1000
#  do
#    python main.py +scenario=cil_cifar100_10 +model="$MODEL" +training=cifar10_5 +method=oewc_default optimizer=adam  device="$DEVICE" method.ewc_lambda=$lambda experiment=dev hydra=search
#  done
#;;
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
#              python main.py +scenario=cil_cifar100_10 +model=$MODEL +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=dev method.mem_size=$memory method.margin=$margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=0.5 method.gamma=$gamma hydra=search
#            done
#          done
#        done
#      done
#    done
#;;
#routing)
##  routing_3l_convblock
##  routing_3l_convblock_invusage
#  for memory in 200 500 1000 2000
#  do
#    for margin in 1 0.3 0.2
#     do
#      for past_margin_w in 0.5 0.1 0.01
#      do
#        for future_margin in 5
#        do
#            for gamma in 1
#            do
#                for pa in random usage inverse_usage
#                do
#                for pt in task
#                do
#              python main.py +scenario=cil_cifar100_10 +model=$MODEL +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=dev method.mem_size=$memory method.margin=$margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=1 method.gamma=$gamma hydra=search model.path_selection_strategy=$pa model.prediction_mode=$pt wandb_prefix=routing_model_search_
#            done
#            done
#            done
#          done
#        done
#      done
#    done
#;;

#icarl)
#  python main.py +scenario=cil_cifar100_10 +model=resnet20 +training=cifar10_5 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/icarl/icarl_2000/'
#;;
#replay)
#  python main.py +scenario=cil_cifar100_10 +model=resnet20 +training=cifar10_5 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/replay/replay_500'
#;;
#cope)
#  python main.py +scenario=cil_cifar100_10 +model=resnet20 +training=cifar10_5 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
