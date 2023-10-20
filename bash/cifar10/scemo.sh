#!/usr/bin/env bash

#python main.py +scenario=class_incremental_cifar10 +model=routing_3l_convblock +training=cifar10 +method=routing_cifar10 optimizer=adam device=0 method.mem_size=200 model.path_selection_strategy=usage

python main.py +scenario=class_incremental_cifar10 +model=routing_3l_doubleconvblock +training=cifar10 +method=routing_cifar10 optimizer=adam device=0 method.mem_size=200 model.path_selection_strategy=inverse_usage
python main.py +scenario=class_incremental_cifar10 +model=routing_3l_doubleconvblock +training=cifar10 +method=routing_cifar10 optimizer=adam device=0 method.mem_size=200 model.path_selection_strategy=usage
