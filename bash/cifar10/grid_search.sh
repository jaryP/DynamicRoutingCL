#!/usr/bin/env bash
METHOD=$1
MODEL=$2
DEVICE=$3

case $METHOD in
der)
  python main.py --multirun +scenario=class_incremental_cifar10 +model="$MODEL" +training=cifar10 +method=der_default optimizer=adam device=$DEVICE +model.head_classes=100 method.alpha=0.1,0.2,0.5,0.8,1.0 method.beta=0.1,0.2,0.5,0.8,1.0 experiment=dev
;;
ewc)
  python main.py --multirun +scenario=class_incremental_cifar10 +model="$MODEL" +training=cifar10 +method=ewc_default optimizer=adam  device="$DEVICE" method.lambda=1,10,100,1000 experiment=dev
;;
#oewc)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=oewc_100 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/oewc/oewc_100'
#;;
#cml)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cml/cml_01'
#;;
#naive)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10  +method=naive optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/naive/'
#;;
#cumulative)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cumulative optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cumulative/'
#;;
#icarl)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=icarl_2000 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/icarl/icarl_2000/'
#;;
#replay)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/replay/replay_500'
#;;
#ssil)
#python main.py +scenario=class_incremental_cifar10 +model="$MODEL" +training=cifar10 +method=ssil_200 optimizer=adam  device="$DEVICE"
#python main.py +scenario=class_incremental_cifar10 +model="$MODEL" +training=cifar10 +method=ssil_500 optimizer=adam  device="$DEVICE"
#python main.py +scenario=class_incremental_cifar10 +model="$MODEL" +training=cifar10 +method=ssil_1000 optimizer=adam  device="$DEVICE"
#python main.py +scenario=class_incremental_cifar10 +model="$MODEL" +training=cifar10 +method=ssil_2000 optimizer=adam  device="$DEVICE"
#;;
#cope)
#  python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/ci_cifar10/resnet20/cope/cope_500'
#;;
*)
  echo -n "Unrecognized method"
esac
