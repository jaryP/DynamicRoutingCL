#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  for memory in 200 500 1000 2000 5000
  do
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=gem method.patterns_per_exp=$memory  optimizer=sgd  device="$DEVICE"
  done
;;
#ewc)
#  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=ewc_100 optimizer=sgd  device="$DEVICE"
#;;
#oewc)
#  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=oewc_100 optimizer=sgd  device="$DEVICE"
#;;
#cml)
#  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=cml_01 optimizer=sgd  device="$DEVICE"
#;;
#margin)
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=200 method.past_task_reg=0.1 method.gamma=1 +method.alpha=0 +wadnb_tags=[final_results_margin] method.margin_type=adaptive +method.rehearsal_metric='kl'
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=500 method.past_task_reg=0.5 method.gamma=1 +method.alpha=0 +wadnb_tags=[final_results_margin] method.margin_type=adaptive +method.rehearsal_metric='kl'
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=1000 method.past_task_reg=0.25 method.gamma=1 +method.alpha=0 +wadnb_tags=[final_results_margin] method.margin_type=adaptive +method.rehearsal_metric='kl'
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=2000 method.past_task_reg=1 method.gamma=1 +method.alpha=0 +wadnb_tags=[final_results_margin] method.margin_type=adaptive +method.rehearsal_metric='kl'
#;;
#der)
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der_200  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8 head=linear +wadnb_tags=[final_results]
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der_500  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=1 head=linear +wadnb_tags=[final_results]
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der_1000  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8 head=linear +wadnb_tags=[final_results]
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der_2000  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8 head=linear +wadnb_tags=[final_results]
#;;
naive)
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet  +method=naive optimizer=sgd  device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=cumulative optimizer=sgd  device="$DEVICE"
;;
#icarl)
#  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=icarl_2000 optimizer=sgd  device="$DEVICE"
#;;
replay)
  for memory in 200 500 1000 2000 5000
  do
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=replay optimizer=sgd method.mem_size=$memory  device="$DEVICE"
  done
;;
rpc)
  for memory in 200 500 1000 2000 5000
  do
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=replay optimizer=sgd method.mem_size=$memory device="$DEVICE" head=simplex
  done
;;
ssil)
  for memory in 200 500 1000 2000
  do
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=ssil optimizer=sgd method.mem_size=$memory device="$DEVICE" training.epochs=100
  done
;;
er_ace)
  for memory in 200 500 1000 2000 5000
  do
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=er_ace method.mem_size=$memory optimizer=sgd  device="$DEVICE"
  done
;;
gdumb)
  for memory in 200 500 1000 2000 5000
  do
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=gdumb method.mem_size=$memory optimizer=sgd  device="$DEVICE"
  done
;;
#logit_d)
#  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=cifar100 device=$DEVICE method.mem_size=200 method.alpha=1 head=incremental
#  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=cifar100 device=$DEVICE method.mem_size=500 method.alpha=1 head=incremental
#  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=cifar100 device=$DEVICE method.mem_size=1000 method.alpha=0.75 head=incremental
#  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=cifar100 device=$DEVICE method.mem_size=2000 method.alpha=0.75 head=incremental
#;;
*)
  echo -n "Unrecognized method"
esac

#2107689 2466435
