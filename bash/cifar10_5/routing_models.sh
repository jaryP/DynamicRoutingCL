#!/usr/bin/env bash

DEVICE=$1

#for memory in 200 500 1000 2000
#do
#  for pa in random usage inverse_usage
#  do
#    for pt in task class
#    do
#      python main.py +scenario=cil_cifar10_5 +model=routing_5l_convblock_p model.path_selection_strategy=$pa model.prediction_mode=$pt +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=dev method.mem_size=$memory hydra=search wandb_prefix=routing_model_search_
#    done
#  done
#done

for memory in 200 500 1000 2000
do
  for past_margin in 1 0.5 0.3 0.2
   do
    for past_margin_w in 0.1 0.01 0.01 0.001
    do
      for future_margin in 5
      do
          for gamma in 1 0.8 0.6 0.5
          do
            python main.py +scenario=cil_cifar10_5 +model=routing_5l_convblock_p +training=cifar10_5 +method=routing_cifar10 +method.reg_sampling_bs=-1 training.epochs=20 optimizer=adam device=$DEVICE method.mem_size=$memory method.past_margin=$past_margin  method.future_margin=$future_margin method.past_task_reg=$past_margin_w method.future_task_reg=1 method.gamma=$gamma hydra=search experiment=base1 wandb_prefix=lenovo_ +wadnb_tags=search
          done
        done
      done
    done
  done