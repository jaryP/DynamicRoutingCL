#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gem_200   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gem_500   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gem_1000   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gem_2000   device="$DEVICE"
;;
#ewc)
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=ewc_100   device="$DEVICE"
#;;
#oewc)
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=oewc_100   device="$DEVICE"
#;;
#cml)
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=cml_01   device="$DEVICE"
#;;
naive)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5  +method=naive   device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=cumulative   device="$DEVICE"
;;
#icarl)
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=icarl_2000   device="$DEVICE"
#;;
replay)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=replay_200 optimizer=sgd  device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=replay_500 optimizer=sgd  device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=replay_1000 optimizer=sgd  device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=replay_2000 optimizer=sgd  device="$DEVICE"
;;
ssil)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=ssil_200   device="$DEVICE" training.epochs=100
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=ssil_500   device="$DEVICE" training.epochs=100
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=ssil_1000   device="$DEVICE" training.epochs=100
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=ssil_2000   device="$DEVICE" training.epochs=100
;;
er_ace)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=200   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=500   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=1000   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=2000   device="$DEVICE"
;;
gdumb)
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=200   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=500   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=1000   device="$DEVICE"
  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=2000   device="$DEVICE"
;;
#der)
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=der_200  device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=1
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=der_500  device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=0.8
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=der_1000  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=der_2000  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=1
#;;
#routing)
#  python main.py +scenario=cil_cifar10_5 +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.01 method.past_margin=1 method.gamma=1
#  python main.py +scenario=cil_cifar10_5 +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.01 method.past_margin=0.3 method.mem_size=500 method.gamma=1
#  python main.py +scenario=cil_cifar10_5 +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.01 method.past_margin=0.3 method.mem_size=1000 method.gamma=1
#  python main.py +scenario=cil_cifar10_5 +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.1 method.past_margin=1 method.mem_size=2000 method.gamma=1
#;;
#cope)
#  python main.py +scenario=cil_cifar10_5 +model="$MODEL" +training=cifar10_5 +method=cope   device="$DEVICE" hydra.run.dir='./results/ci_cifar10/"$MODEL"/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
