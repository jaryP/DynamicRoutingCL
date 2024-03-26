#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
  for memory in  500 1000 2000 5000
  do
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=gem method.patterns_per_exp=$memory  optimizer=sgd  device="$DEVICE"
  done
;;
margin)
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size= method.past_task_reg=0.05 method.gamma=1 +wadnb_tags=[final_results_margin] method.margin_type=adaptive
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=500 method.past_task_reg=0.05 method.gamma=1 +wadnb_tags=[final_results_margin] method.margin_type=adaptive
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=1000 method.past_task_reg=0.05 method.gamma=1 +wadnb_tags=[final_results_margin] method.margin_type=adaptive
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=2000 method.past_task_reg=0.25 method.gamma=1 +wadnb_tags=[final_results_margin] method.margin_type=adaptive
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=margin head=margin_head device=$DEVICE method.mem_size=5000 method.past_task_reg=0.25 method.gamma=1 +wadnb_tags=[final_results_margin] method.margin_type=adaptive
;;
der)
#python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der  device=$DEVICE method.mem_size=  method.alpha=0.1 method.beta=0.5 head=linear +wadnb_tags=[final_results]
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der  device=$DEVICE method.mem_size=500  method.alpha=0.1 method.beta=0.1 head=linear +wadnb_tags=[final_results]
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der  device=$DEVICE method.mem_size=1000 method.alpha=0.1 method.beta=0.5 head=linear +wadnb_tags=[final_results]
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der  device=$DEVICE method.mem_size=2000  method.alpha=0.1 method.beta=0.2 head=linear +wadnb_tags=[final_results]
python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=der  device=$DEVICE method.mem_size=5000 method.alpha=0.1 method.beta=0.2 head=linear +wadnb_tags=[final_results]
;;
naive)
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet  +method=naive optimizer=sgd  device="$DEVICE"
;;
cumulative)
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=cumulative optimizer=sgd  device="$DEVICE"
;;
replay)
  for memory in  500 1000 2000 5000
  do
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=replay optimizer=sgd method.mem_size=$memory  device="$DEVICE"
  done
;;
rpc)
  for memory in  500 1000 2000 5000
  do
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=replay optimizer=sgd method.mem_size=$memory device="$DEVICE" head=simplex
  done
;;
ssil)
  for memory in  500 1000 2000 5000
  do
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=ssil optimizer=sgd method.mem_size=$memory device="$DEVICE" training.epochs=30
  done
;;
er_ace)
  for memory in  500 1000 2000 5000
  do
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=er_ace method.mem_size=$memory optimizer=sgd  device="$DEVICE"
  done
;;
lode)
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=lode method.mem_size=500 method.rho=0.1 device="$DEVICE"
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=lode method.mem_size=1000 method.rho=0.1 device="$DEVICE"
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=lode method.mem_size=2000 method.rho=0.2 device="$DEVICE"
  python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=lode method.mem_size=5000 method.rho=0.5 device="$DEVICE"
;;
gdumb)
  for memory in  500 1000 2000 5000
  do
    python main.py +scenario=cil_tyn_10 model="$MODEL" +training=tinyimagenet +method=gdumb method.mem_size=$memory optimizer=sgd  device="$DEVICE"
  done
;;
logit_d)
#  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=logit_d device=$DEVICE method.mem_size=200 method.alpha=1 head=incremental
  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=logit_d device=$DEVICE method.mem_size=500 method.alpha=0.5 head=incremental
  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=logit_d device=$DEVICE method.mem_size=1000 method.alpha=1 head=incremental
  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=logit_d device=$DEVICE method.mem_size=2000 method.alpha=1 head=incremental
  python main.py +scenario=cil_tyn_10 model=$MODEL +training=tinyimagenet +method=logit_d device=$DEVICE method.mem_size=5000 method.alpha=0.75 head=incremental
;;
*)
  echo -n "Unrecognized method"
esac
