#!/usr/bin/env bash

METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
gem)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gem_200   device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gem_500   device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gem_1000   device="$DEVICE" experiment=base10
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gem_2000   device="$DEVICE" experiment=base10
;;
naive)
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5  +method=naive   device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
;;
cumulative)
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=cumulative   device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
;;
#icarl)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=icarl_2000   device="$DEVICE" experiment=base10
#;;
replay)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=replay_200 optimizer=sgd  device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=replay_500 optimizer=sgd  device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=replay_1000 optimizer=sgd  device="$DEVICE" experiment=base10
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=replay_2000 optimizer=sgd  device="$DEVICE" experiment=base10
;;
ssil)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=ssil_200   device="$DEVICE" experiment=base10 training.epochs=100
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=ssil_500   device="$DEVICE" experiment=base10 training.epochs=100 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=ssil_1000   device="$DEVICE" experiment=base10 training.epochs=100
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=ssil_2000   device="$DEVICE" experiment=base10 training.epochs=100
;;
er_ece)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=200   device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=500   device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=1000   device="$DEVICE" experiment=base10
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=er_ece_default method.mem_size=2000   device="$DEVICE" experiment=base10
;;
gdumb)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=200   device="$DEVICE" experiment=base10
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=500   device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=1000   device="$DEVICE" experiment=base10
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=gdumb method.mem_size=2000   device="$DEVICE" experiment=base10
;;
der)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=der_200  device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=1 experiment=base10
  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=der_500  device=$DEVICE +model.head_classes=100 method.alpha=0.2 method.beta=0.8  experiment=base10 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=der_1000  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=0.8  experiment=base10
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=der_2000  device=$DEVICE +model.head_classes=100 method.alpha=0.1 method.beta=1  experiment=base10
;;
margin)
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=200 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.05  device="$DEVICE" optimizer.momentum=0.9
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=200 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.1 +method.rehearsal_metric='kl' method.margin_type=adaptive device="$DEVICE" +wadnb_tags=[final_results_margin]
  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=500 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.25 +method.rehearsal_metric='kl' method.margin_type=adaptive  device="$DEVICE" experiment=base10 +wadnb_tags=[final_results_rp]
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=1000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.25 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[final_results_margin]
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=2000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.5 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[final_results_margin]
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=500   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=1000   device="$DEVICE"
#  python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=er_ace_default method.mem_size=2000   device="$DEVICE"
;;
#routing)
#  python main.py +scenario=cil_cifar10_rp model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.01 method.margin=1 method.gamma=1  experiment=base10
#  python main.py +scenario=cil_cifar10_rp model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.01 method.margin=0.3 method.mem_size=500 method.gamma=1  experiment=base10
#  python main.py +scenario=cil_cifar10_rp model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.01 method.margin=0.3 method.mem_size=1000 method.gamma=1  experiment=base10
#  python main.py +scenario=cil_cifar10_rp model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10  device=$DEVICE method.past_task_reg=0.1 method.margin=1 method.mem_size=2000 method.gamma=1  experiment=base10
#;;
#cope)
#  python main.py +scenario=cil_cifar10_rp model="$MODEL" +training=cifar10_5 +method=cope   device="$DEVICE" experiment=base10 hydra.run.dir='./results/ci_cifar10/"$MODEL"/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
