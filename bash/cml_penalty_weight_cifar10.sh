#!/usr/bin/env bash

DEVICE=$1

#CM
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ci_cifar10/resnet20/cml/cml_100' experiment.experiments=1 method.penalty_weight=100 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ci_cifar10/resnet20/cml/cml_10' experiment.experiments=1 method.penalty_weight=10 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ci_cifar10/resnet20/cml/cml_1' experiment.experiments=1 method.penalty_weight=1 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ci_cifar10/resnet20/cml/cml_01' experiment.experiments=1 method.penalty_weight=0.1 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ci_cifar10/resnet20/cml/cml_001' experiment.experiments=1 method.penalty_weight=0.01 +scenario.shuffle_first=true experiment.load=false

python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ti_cifar10/resnet20/cml/cml_100' experiment.experiments=1 method.penalty_weight=100 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ti_cifar10/resnet20/cml/cml_10' experiment.experiments=1 method.penalty_weight=10 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ti_cifar10/resnet20/cml/cml_1' experiment.experiments=1 method.penalty_weight=1 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ti_cifar10/resnet20/cml/cml_01' experiment.experiments=1 method.penalty_weight=0.1 +scenario.shuffle_first=true experiment.load=false
python main.py +scenario=task_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd training.device="$DEVICE" hydra.run.dir='./results/ablation/penalty_w/ti_cifar10/resnet20/cml/cml_001' experiment.experiments=1 method.penalty_weight=0.01 +scenario.shuffle_first=true experiment.load=false
