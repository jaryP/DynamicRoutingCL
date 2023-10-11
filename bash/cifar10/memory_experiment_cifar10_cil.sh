#!/usr/bin/env bash

DEVICE=$1

# CM
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd +method.proj_w=1 training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cml/cml_100' experiment.experiments=1 method.memory_parameters.memory_size=100 +scenario.shuffle_first=true experiment.load=false
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd +method.proj_w=1 training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cml/cml_250' experiment.experiments=1 method.memory_parameters.memory_size=250 +scenario.shuffle_first=true experiment.load=false
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd +method.proj_w=1 training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cml/cml_500' experiment.experiments=1 method.memory_parameters.memory_size=500 +scenario.shuffle_first=true experiment.load=false
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd +method.proj_w=1 training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cml/cml_1000' experiment.experiments=1 method.memory_parameters.memory_size=1000 +scenario.shuffle_first=true experiment.load=false
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd +method.proj_w=1 training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cml/cml_1500' experiment.experiments=1 method.memory_parameters.memory_size=1500 +scenario.shuffle_first=true experiment.load=false
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cml_01 optimizer=sgd +method.proj_w=1 training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cml/cml_2000' experiment.experiments=1 method.memory_parameters.memory_size=2000 +scenario.shuffle_first=true experiment.load=false
#
## REPLAY
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/replay/replay_100' method.mem_size=100 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/replay/replay_250' method.mem_size=250 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/replay/replay_500' method.mem_size=500 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/replay/replay_1000' method.mem_size=1000 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/replay/replay_1500' method.mem_size=1500 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=replay_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/replay/replay_2000' method.mem_size=2000 experiment.experiments=1
#
## GEM
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/gem/gem_100' method.patterns_per_experience=20 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/gem/gem_250' method.patterns_per_experience=50 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/gem/gem_500' method.patterns_per_experience=100 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/gem/gem_1000' method.patterns_per_experience=200 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/gem/gem_1500' method.patterns_per_experience=300 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=gem_500 optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/gem/gem_2000' method.patterns_per_experience=500 experiment.experiments=1

# SSIL
python main.py +scenario=class_incremental_cifar10 +model=alexnet +training=cifar10 +method=ssil_200 optimizer=adam  device="$DEVICE"
python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=ssil_500 optimizer=adam  device="$DEVICE"
python main.py +scenario=class_incremental_cifar10 +model=alexnet +training=cifar10 +method=ssil_1000 optimizer=adam  device="$DEVICE"
python main.py +scenario=class_incremental_cifar10 +model=alexnet +training=cifar10 +method=ssil_2000 optimizer=adam  device="$DEVICE"

# COPE
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cope_review/cope_100' method.memory_size=20 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cope_review/cope_250' method.memory_size=50 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cope_review/cope_500' method.memory_size=100 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cope_review/cope_1000' method.memory_size=200 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cope_review/cope_1500' method.memory_size=300 experiment.experiments=1
#python main.py +scenario=class_incremental_cifar10 +model=resnet20 +training=cifar10 +method=cope optimizer=sgd  training.device="$DEVICE" hydra.run.dir='./results/memory_experiment/ci_cifar10/resnet20/cope_review/cope_2000' method.memory_size=500 experiment.experiments=1
