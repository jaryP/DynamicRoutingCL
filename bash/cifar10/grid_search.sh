#!/usr/bin/env bash
METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
der)
  for memory in 200 500 1000 2000
    do
    for alpha in 0.1 0.2 0.5 0.8 1.0
    do
      for beta in 0.1 0.2 0.5 0.8 1.0
      do
        python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=der_default optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=$alpha method.beta=$beta experiment=dev method.mem_size=$memory hydra=search
      done
    done
  done
;;
ewc)
  for lambda in 1 10 100 1000
  do
    python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=ewc_default optimizer=adam  device="$DEVICE" method.lambda=$lambda experiment=dev
  done
;;
oewc)
  for lambda in 1 10 100 1000
  do
    python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=oewc_default optimizer=adam  device="$DEVICE" method.lambda=$lambda experiment=dev
  done
;;
routing)
  for memory in 200 500 1000 2000
  do
    for past_margin in 0.1 0.2 0.3 0.5
    do
      for past_margin_w in 1 0.5 0.1 0.01 0.001
      do
        for future_margin in 1 2 3 4 5
        do
            for gamma in 1 0.5 0.1
            do
              python main.py +scenario=cil_cifar10_5tasks +model=routing_tiny_rt +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=dev method.memory_size=$memory method.past_margin=$past_margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=0.5 method.gamma=$gamma hydra=search
            done
          done
        done
      done
    done
;;
#oewc)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/oewc/oewc_100'
#;;
#cml)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cml/cml_01'
#;;
#naive)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/naive/'
#;;
#cumulative)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cumulative/'
#;;
#icarl)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/icarl/icarl_2000/'
#;;
#replay)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/replay/replay_500'
#;;
#ssil)
#python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=ssil_200 optimizer=adam  device="$DEVICE"
#python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=ssil_500 optimizer=adam  device="$DEVICE"
#python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=ssil_1000 optimizer=adam  device="$DEVICE"
#python main.py +scenario=cil_cifar10_5tasks +model="$MODEL" +training=cifar10_5 +method=ssil_2000 optimizer=adam  device="$DEVICE"
#;;
#cope)
#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
