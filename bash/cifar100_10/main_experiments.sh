#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  for memory in 200 500 1000 2000
  do
    python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gem method.patterns_per_exp=$memory  optimizer=sgd experiment=base2  device="$DEVICE"
  done
;;
lode)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=lode method.mem_size=200 method.rho=0.1 device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=lode method.mem_size=500 method.rho=0.2 device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=lode method.mem_size=1000 method.rho=0.2 device="$DEVICE"
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=lode method.mem_size=2000 method.rho=0.5 device="$DEVICE"
;;
margin)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=margin head=margin_head device=$DEVICE method.mem_size=200 method.past_task_reg=0.1 method.gamma=1  +wadnb_tags=[final_results_margin] method.margin_type=adaptive
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=margin head=margin_head device=$DEVICE method.mem_size=500 method.past_task_reg=0.5 method.gamma=1  +wadnb_tags=[final_results_margin] method.margin_type=adaptive
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=margin head=margin_head device=$DEVICE method.mem_size=1000 method.past_task_reg=0.25 method.gamma=1  +wadnb_tags=[final_results_margin] method.margin_type=adaptive
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=margin head=margin_head device=$DEVICE method.mem_size=2000 method.past_task_reg=1 method.gamma=1  +wadnb_tags=[final_results_margin] method.margin_type=adaptive
;;
der)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=der  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8 head=linear +wadnb_tags=[final_results]
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=der  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=1 head=linear +wadnb_tags=[final_results]
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=der  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8 head=linear +wadnb_tags=[final_results]
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=der  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8 head=linear +wadnb_tags=[final_results]
;;
naive)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100  +method=naive optimizer=sgd  device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=cumulative optimizer=sgd  device="$DEVICE"
;;
#icarl)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=icarl_2000 optimizer=sgd  device="$DEVICE"
#;;
replay)
  for memory in 200 500 1000 2000
  do
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=replay optimizer=sgd method.mem_size=$memory  device="$DEVICE"
  done
;;
rpc)
  for memory in 200 500 1000 2000
  do
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=replay optimizer=sgd method.mem_size=$memory device="$DEVICE" head=simplex
  done
;;
ssil)
  for memory in 200 500 1000 2000
  do
  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=ssil optimizer=sgd method.mem_size=$memory device="$DEVICE" training.epochs=100 experiment=base2
  done
;;
er_ece)
  for memory in 200 500 1000 2000
  do
    python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=er_ace method.mem_size=$memory optimizer=sgd  device="$DEVICE"
  done
;;
gdumb)
  for memory in 200 500 1000 2000
  do
    python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=gdumb method.mem_size=$memory optimizer=sgd  device="$DEVICE"
  done
;;
logit_d)
  python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=logit_d device=$DEVICE method.mem_size=200 method.alpha=1 head=incremental
  python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=logit_d device=$DEVICE method.mem_size=500 method.alpha=1 head=incremental
  python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=logit_d device=$DEVICE method.mem_size=1000 method.alpha=0.75 head=incremental
  python main.py +scenario=cil_cifar100_10 model=$MODEL +training=cifar100 +method=logit_d device=$DEVICE method.mem_size=2000 method.alpha=0.75 head=incremental
;;
#cope)
#  python main.py +scenario=cil_cifar100_10 model="$MODEL" +training=cifar100 +method=cope optimizer=sgd  device="$DEVICE" hydra.run.dir='./results/ci_cifar10/"$MODEL"/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
