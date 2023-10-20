#!/usr/bin/env bash

#!/usr/bin/env bash
METHOD=$1
DEVICE=$2

case $METHOD in
der)
  python main.py +scenario=cil_cifar10_5tasks +model=alexnet +training=cifar10_5 +method=der_default optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=1 method.beta=1 experiment=base1 method.mem_size=200
  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=der_default optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=1 method.beta=1 experiment=base1 method.mem_size=200
;;
replay)
  python main.py +scenario=cil_cifar10_5tasks +model=alexnet +training=cifar10_5 +method=replay_default optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200
  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=replay_default optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200
;;
routing)

  python main.py +scenario=cil_cifar10_5tasks +model=routing_3l_convblock +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200 model.path_selection_strategy=random
  python main.py +scenario=cil_cifar10_5tasks +model=routing_3l_convblock +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200 model.path_selection_strategy=usage
  python main.py +scenario=cil_cifar10_5tasks +model=routing_3l_convblock +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200 model.path_selection_strategy=inverse_usage

  python main.py +scenario=cil_cifar10_5tasks +model=routing_3l_doubleconvblock +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200 model.path_selection_strategy=random
  python main.py +scenario=cil_cifar10_5tasks +model=routing_3l_doubleconvblock +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200 model.path_selection_strategy=usage
  python main.py +scenario=cil_cifar10_5tasks +model=routing_3l_doubleconvblock +training=cifar10_5 +method=routing_cifar10 optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200 model.path_selection_strategy=inverse_usage

#  python main.py +scenario=cil_cifar10_5tasks +model=resnet20 +training=cifar10_5 +method=replay_default optimizer=adam device=$DEVICE experiment=base1 method.mem_size=200
;;
*)
  echo -n "Unrecognized method"
esac
