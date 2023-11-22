#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gem_200 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gem_500 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gem_1000 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gem_2000 optimizer=adam  device="$DEVICE"
;;
#ewc)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=ewc_100 optimizer=adam  device="$DEVICE"
#;;
#oewc)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=oewc_100 optimizer=adam  device="$DEVICE"
#;;
#cml)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=cml_01 optimizer=adam  device="$DEVICE"
#;;
naive)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100  +method=naive optimizer=adam  device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=cumulative optimizer=adam  device="$DEVICE"
;;
#icarl)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=icarl_2000 optimizer=adam  device="$DEVICE"
#;;
replay)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=replay_200 optimizer=sgd  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=replay_500 optimizer=sgd  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=replay_1000 optimizer=sgd  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=replay_2000 optimizer=sgd  device="$DEVICE"
;;
ssil)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=ssil_200 optimizer=adam  device="$DEVICE" training.epochs=100
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=ssil_500 optimizer=adam  device="$DEVICE" training.epochs=100
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=ssil_1000 optimizer=adam  device="$DEVICE" training.epochs=100
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=ssil_2000 optimizer=adam  device="$DEVICE" training.epochs=100
;;
er_ece)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=er_ece_default method.mem_size=200 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=er_ece_default method.mem_size=500 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=er_ece_default method.mem_size=1000 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=er_ece_default method.mem_size=2000 optimizer=adam  device="$DEVICE"
;;
gdumb)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gdumb method.mem_size=200 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gdumb method.mem_size=500 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gdumb method.mem_size=1000 optimizer=adam  device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gdumb method.mem_size=2000 optimizer=adam  device="$DEVICE"
;;
#cope)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=cope optimizer=adam  device="$DEVICE" hydra.run.dir='./results/ci_cifar10/"$MODEL"/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
