#!/usr/bin/env bash

MODEL=$1
DEVICE=$2

python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=4000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.5 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[ablation_margin] experiment=base1
python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=6000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.5 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[ablation_margin] experiment=base1
python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=8000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.5 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[ablation_margin] experiment=base1
python main.py +scenario=cil_cifar10_5 model="$MODEL" +training=cifar10_5 +method=margin_cifar10 method.mem_size=10000 head=margin_head method.gamma=1 +method.alpha=0 method.past_task_reg=0.5 +method.rehearsal_metric='kl'  device="$DEVICE" +wadnb_tags=[ablation_margin] experiment=base1
