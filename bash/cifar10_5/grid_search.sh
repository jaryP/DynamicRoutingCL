#!/usr/bin/env bash
METHOD=$1
MODEL=$2
MEMORY=$3
DEVICE=$4

case $METHOD in
der)
  for memory in 200 500 1000 2000
    do
    for alpha in 0.1 0.2 0.5 0.8 1.0
    do
      for beta in 0.1 0.2 0.5 0.8 1.0
      do
        python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der_default device=$DEVICE head=linear method.alpha=$alpha method.beta=$beta experiment=dev method.mem_size=$memory hydra=search +wadnb_tags=[grid_search]
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
            python main.py +scenario=cil_cifar10_5 model=$MODEL +training=cifar10_5 +method=margin_cifar10 device=$DEVICE method.mem_size=$memory method.past_task_reg=$past_margin_w method.gamma=$gamma +method.alpha=$alpha hydra=search +wadnb_tags=[grid_search] experiment=dev head=margin_head wandb_prefix=lenovo_
          done
          done
      done
    done
;;

#routing)
##  routing_3l_convblock
##  routing_3l_convblock_invusage
#  for memory in 200 500 1000 2000
#  do
#    for margin in 1 0.5 0.25
#     do
#      for past_margin_w in 0.1 0.01 0.01
#      do
#        for future_margin in 5
#        do
#            for gamma in 1
#            do
#              python main.py +scenario=cil_cifar10_5 +model=$MODEL +training=cifar10_5 +method=routing_cifar10 +method.reg_sampling_bs=-1 training.epochs=20 optimizer=adam device=$DEVICE method.mem_size=$memory method.margin=$margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=1 method.gamma=$gamma hydra=search experiment=base1 wandb_prefix=lenovo_ +wadnb_tags=test
#              python main.py +scenario=cil_cifar10_5 +model=$MODEL +training=cifar10_5 +method=routing_cifar10 +method.reg_sampling_bs=32 training.epochs=50 optimizer=adam device=$DEVICE method.mem_size=$memory method.margin=$margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=1 method.gamma=$gamma hydra=search experiment=base1 wandb_prefix=lenovo_ +wadnb_tags=test
#            done
#          done
#        done
#      done
#    done
#;;

#icarl)
#  python main.py +scenario=cil_cifar10_5 +model=resnet20 +training=cifar10_5 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/icarl/icarl_2000/'
#;;
#replay)
#  python main.py +scenario=cil_cifar10_5 +model=resnet20 +training=cifar10_5 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/replay/replay_500'
#;;
#cope)
#  python main.py +scenario=cil_cifar10_5 +model=resnet20 +training=cifar10_5 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
