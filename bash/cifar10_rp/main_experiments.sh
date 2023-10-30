#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gem_200 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gem_500 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gem_1000 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gem_2000 optimizer=adam  device="$DEVICE" experiment=base10
;;
#ewc)
#  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=ewc_100 optimizer=adam  device="$DEVICE" experiment=base10
#;;
#oewc)
#  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=oewc_100 optimizer=adam  device="$DEVICE" experiment=base10
#;;
#cml)
#  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=cml_01 optimizer=adam  device="$DEVICE" experiment=base10
#;;
naive)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5  +method=naive optimizer=adam  device="$DEVICE" experiment=base10
;;
cumulative)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=cumulative optimizer=adam  device="$DEVICE" experiment=base10
;;
#icarl)
#  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=icarl_2000 optimizer=adam  device="$DEVICE" experiment=base10
#;;
replay)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=replay_200 optimizer=sgd  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=replay_500 optimizer=sgd  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=replay_1000 optimizer=sgd  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=replay_2000 optimizer=sgd  device="$DEVICE" experiment=base10
;;
ssil)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=ssil_200 optimizer=adam  device="$DEVICE" experiment=base10 training.epochs=100
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=ssil_500 optimizer=adam  device="$DEVICE" experiment=base10 training.epochs=100
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=ssil_1000 optimizer=adam  device="$DEVICE" experiment=base10 training.epochs=100
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=ssil_2000 optimizer=adam  device="$DEVICE" experiment=base10 training.epochs=100
;;
er_ece)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=200 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=500 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=1000 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=2000 optimizer=adam  device="$DEVICE" experiment=base10
;;
gdumb)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=200 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=500 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=1000 optimizer=adam  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=2000 optimizer=adam  device="$DEVICE" experiment=base10
;;
der)
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=der_200 optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=1 experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=der_500 optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=0.8  experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=der_1000 optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8  experiment=base10
  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=der_2000 optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=1  experiment=base10
;;
routing)
  python main.py +scenario=cil_cifar10_rp +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE method.past_task_reg=0.01 method.past_margin=1 method.gamma=1  experiment=base10
  python main.py +scenario=cil_cifar10_rp +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE method.past_task_reg=0.01 method.past_margin=0.3 method.mem_size=500 method.gamma=1  experiment=base10
  python main.py +scenario=cil_cifar10_rp +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE method.past_task_reg=0.01 method.past_margin=0.3 method.mem_size=1000 method.gamma=1  experiment=base10
  python main.py +scenario=cil_cifar10_rp +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE method.past_task_reg=0.1 method.past_margin=1 method.mem_size=2000 method.gamma=1  experiment=base10
;;
#cope)
#  python main.py +scenario=cil_cifar10_rp +model="$MODEL" +training=cifar10_5 +method=cope optimizer=adam  device="$DEVICE" experiment=base10 hydra.run.dir='./results/ci_cifar10/"$MODEL"/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
