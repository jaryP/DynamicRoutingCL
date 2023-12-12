#!/usr/bin/env bash

DEVICE=$1

for memory in 200 500 1000 2000
do
  for pa in random usage inverse_usage
  do
    for pt in task class
    do
      python main.py +scenario=cil_cifar100_10 +model=routing_5l_convblock_p model.path_selection_strategy=$pa model.prediction_mode=$pt +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=dev method.mem_size=$memory hydra=search wandb_prefix=routing_model_search_
    done
  done
done