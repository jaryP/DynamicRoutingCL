#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  for memory in +method=margin200 500 1000 2000 
  do
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gem method.patterns_per_exp=$memory device="$DEVICE"
  done
;;
#ewc)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ewc_100   device="$DEVICE"
#;;
#oewc)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=oewc_100   device="$DEVICE"
#;;
#cml)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=cml_01   device="$DEVICE"
#;;
naive)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5  +method=naive   device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=cumulative   device="$DEVICE"
;;
icarl)
  for memory in +method=margin200 500 1000 2000 
  do
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=icarl method.memory_size=$memory  device="$DEVICE"
  done
;;
replay)
  for memory in +method=margin200 500 1000 2000 
  do
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=replay method.memory_size=$memory device="$DEVICE"
  done
;;
rpc)
  for memory in +method=margin200 500 1000 2000 
  do
      python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=replay method.memory_size=$memory device="$DEVICE" head=simplex
  done
;;
margin)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin method.mem_size=200 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.1 +method.rehearsal_metric='kl' method.margin_type=adaptive device="$DEVICE" +wadnb_tags=[final_results_margin]
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin method.mem_size=500 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.25 +method.rehearsal_metric='kl' method.margin_type=adaptive  device="$DEVICE" +wadnb_tags=[final_results_margin]
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin method.mem_size=1000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.25 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[final_results_margin]
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin method.mem_size=2000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.5 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[final_results_margin]
;;
ssil)
  for memory in +method=margin200 500 1000 2000 
  do
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=ssil method.mem_size=$memory  device="$DEVICE" training.epochs=100
  done 
;;
er_ace)
  for memory in +method=margin200 500 1000 2000 
  do
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace method.mem_size=$memory   device="$DEVICE"
  done 
;;
gdumb)
  for memory in +method=margin200 500 1000 2000 
  do
    python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=$memory device="$DEVICE"
  done
;;
der)
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der  method.mem_size=200 device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.5 head=linear
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der method.mem_size=500  device=$DEVICE +model.head_classes=100 method.alpha=0.5 method.beta=0.8 head=linear
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der method.mem_size=1000  device=$DEVICE +model.head_classes=100 method.alpha=0.5 method.beta=0.1 head=linear
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=der method.mem_size=2000  device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=0.8 head=linear
;;
#cope)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=cope   device="$DEVICE" hydra.run.dir='./results/ci_cifar10/"$MODEL"/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
