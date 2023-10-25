import json
import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from avalanche.benchmarks import data_incremental_benchmark
from avalanche.logging import WandBLogger

from avalanche.training import DER
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss

from base.scenario import get_dataset_nc_scenario
from models.base import get_cl_model


def process_saving_path(sstring: str) -> str:
    return '__'.join(s.split('=')[1] for s in sstring.split('__'))


OmegaConf.register_new_resolver("process_saving_path", process_saving_path)
OmegaConf.register_new_resolver("method_path", lambda x: x.split('.')[-1])


@hydra.main(config_path="configs",
            version_base='1.1',
            config_name="config")
def avalanche_training(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(os.getcwd())

    # check config

    device = cfg.get('device', 'cpu')

    scenario = cfg['scenario']
    dataset = scenario['dataset']
    scenario_name = scenario['scenario']
    n_tasks = scenario['n_tasks']
    task_incremental_learning = scenario['return_task_id']

    shuffle = scenario['shuffle']
    shuffle_first = scenario.get('shuffle_first', False)

    model_cfg = cfg['model']

    plugin_name = cfg.trainer_name.lower()

    training = cfg['training']
    epochs = training['epochs']
    batch_size = training['batch_size']

    num_workers = training.get('num_workers', 0)
    pin_memory = training.get('pin_memory', True)

    experiment = cfg['experiment']
    n_experiments = experiment.get('experiments', 1)
    load = experiment.get('load', True)

    save = experiment.get('save', True)
    plot = experiment.get('plot', False)
    eval_every = experiment.get('eval_every', 5)

    save_states = experiment.get('save_states', False)
    console_log = experiment.get('console_log', False)

    if device == 'cpu':
        warnings.warn("Device set to cpu.")
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn(f"Device not found {device} "
                      f"or CUDA {torch.cuda.is_available()}")

    device = torch.device(device)

    all_results = []

    base_path = os.getcwd()
    cfg_path = os.path.join(os.getcwd(), '.hydra', 'config.yaml')

    # if os.path.exists(cfg_path):
    #     with open(cfg_path, 'r') as f:
    #         past_cfg = yaml.safe_load(f)
    #
    #         keys_set = set([k for k in past_cfg.keys() if k != 'device'])
    #         b = keys_set == set([k for k in cfg.keys() if k != 'device'])
    #
    #         assert b, (
    #             f'You are launching an experiment having output path {os.getcwd()}, '
    #             f'but the experiment config and the one saved in the folder have not the same keys.')
    #
    #         b = all([past_cfg[k] == cfg[k] for k in keys_set])
    #
    #         assert b, (
    #             f'You are launching an experiment having output path {os.getcwd()}, '
    #             f'but the experiment config and the one saved in the folder not equal: {past_cfg - cfg}')

    is_cil = not task_incremental_learning
    task_incremental_learning = task_incremental_learning if plugin_name != 'cml' \
        else True

    force_sit = True if plugin_name == 'der' else False
    force_sit = False

    head_classes = cfg.model.get('head_classes', None)

    for exp_n in range(1, n_experiments + 1):
        log.info(f'Starting experiment {exp_n} (of {n_experiments})')

        seed = exp_n - 1

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        experiment_path = os.path.join(base_path, str(seed))

        os.makedirs(experiment_path, exist_ok=True)

        results_path = os.path.join(experiment_path, 'last_results.json')
        complete_results_path = os.path.join(experiment_path,
                                             'complete_results.json')
        complete_results_dev_path = os.path.join(experiment_path,
                                                 'complete_results_dev.json')

        train_results_path = os.path.join(experiment_path, 'train_results.json')

        if plugin_name in ['icarl', 'cope', 'ssil', 'moe'] and not is_cil:
            assert is_cil, 'ICarL , CoPE, and ssil only work under Class Incremental Scenario'

        if plugin_name in ['er'] and is_cil:
            assert is_cil, 'ER only work under Task Incremental Scenario'

        tasks = get_dataset_nc_scenario(name=dataset, n_tasks=n_tasks,
                                        til=task_incremental_learning,
                                        shuffle=shuffle_first if exp_n == 0 else shuffle,
                                        seed=seed, force_sit=force_sit,
                                        method_name=plugin_name,
                                        dev_split=training.get('dev_split',
                                                               None))

        log.info(f'Original classes: {tasks.classes_order_original_ids}')
        log.info(f'Original classes per exp: {tasks.original_classes_in_exp}')

        if load and os.path.exists(results_path):
            print('model loaded')
            log.info(f'Results loaded')
            with open(results_path) as json_file:
                results_after_each_task = json.load(json_file)

            if os.path.exists(train_results_path):
                with open(train_results_path) as json_file:
                    train_res = json.load(json_file)

            if os.path.exists(results_path):
                with open(results_path, 'r') as json_file:
                    last_results = json.load(json_file)

                    log.info(last_results)

            else:
                train_res = results_after_each_task
        else:

            img, _, _ = tasks.train_stream[0].dataset[0]

            if plugin_name == 'der':
                assert 'head_classes' in model_cfg, (
                    'Whn using DER you must specify '
                    'the head dimension of the model, '
                    'by setting head_classes parameter'
                    'in the config file.')

            if head_classes is not None:
                del cfg.model.head_classes

            backbone = hydra.utils.instantiate(cfg.model)

            model = get_cl_model(backbone=backbone,
                                 model_name='',
                                 input_shape=tuple(img.shape),
                                 method_name=plugin_name,
                                 head_classes=head_classes,
                                 is_class_incremental_learning=is_cil,
                                 **model_cfg)

            wandb_prefix = cfg.get('wandb_prefix', '')

            wandb_name = f'{cfg.scenario.dataset}/{cfg.scenario.n_tasks}_{cfg.trainer_name}_{backbone.__class__.__name__}_{exp_n}'
            if wandb_prefix is not None and wandb_prefix != '':
                wandb_name = wandb_prefix + wandb_name

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb_dict['saving_path'] = experiment_path

            metrics = hydra.utils.instantiate(cfg.evaluation.metrics)

            loggers = []
            for ev in cfg.evaluation.loggers:
                if 'WandBLogger' in ev['_target_']:
                    v = WandBLogger(project_name=cfg.core.project_name,
                                    run_name=wandb_name,
                                    params={'config': wandb_dict})
                else:
                    v = hydra.utils.instantiate(ev)
                loggers.append(v)

            eval_plugin = EvaluationPlugin(
                metrics,
                loggers=loggers,
                strict_checks=False
            )

            opt = hydra.utils.instantiate(cfg.optimizer,
                                          params=model.parameters())

            criterion = CrossEntropyLoss()

            if cfg is not None and '_target_' in cfg.method:
                if plugin_name == 'ICaRL':
                    strategy = hydra.utils.instantiate(cfg.method,
                                                       feature_extractor=model.feature_extractor,
                                                       classifier=model.classifier,
                                                       optimizer=opt,
                                                       train_epochs=epochs,
                                                       train_mb_size=batch_size,
                                                       evaluator=eval_plugin,
                                                       device=device,
                                                       eval_every=eval_every)
                else:
                    strategy = hydra.utils.instantiate(cfg.method,
                                                       model=model,
                                                       criterion=criterion,
                                                       optimizer=opt,
                                                       train_epochs=epochs,
                                                       train_mb_size=batch_size,
                                                       evaluator=eval_plugin,
                                                       device=device,
                                                       eval_every=eval_every)
            else:
                assert False, f'Method not implemented yet {cfg}'

            if isinstance(strategy, DER) and head_classes is None:
                assert 'head_classes' in cfg.model, (
                    'When using DER strategy, '
                    'parameter model.head_classes must be specified.')

            # trainer = get_trainer(**method,
            #                       tasks=tasks,
            #                       sit=is_cil,
            #                       cfg=cfg)
            #
            # strategy = trainer(model=model,
            #                    criterion=criterion,
            #                    optimizer=opt,
            #                    train_epochs=epochs
            #                    if method['name'].lower() != 'cope' else 1,
            #                    train_mb_size=batch_size,
            #                    eval_every=eval_every,
            #                    evaluator=eval_plugin,
            #                    device=device)

            results_after_each_task = []
            all_results = {}
            all_results_dev = {}

            indexes = np.arange(len(tasks.train_stream))

            if plugin_name == 'cope':
                tasks = data_incremental_benchmark(tasks, batch_size,
                                                   shuffle=True)
                indexes = np.arange(len(tasks.train_stream))

                if cfg.method.get('shuffle', False):
                    np.random.shuffle(indexes)

                for _ in range(epochs):
                    for i in indexes:
                        # for i, experience in enumerate(tasks.train_stream):
                        experience = tasks.train_stream[i]
                        train_res = strategy.train(experiences=experience,
                                                   pin_memory=pin_memory,
                                                   num_workers=num_workers)

                        break

                    results_after_each_task.append(
                        strategy.eval(tasks.test_stream,
                                      pin_memory=pin_memory,
                                      num_workers=num_workers))

            else:
                for i in indexes:
                    # for i, experience in enumerate(tasks.train_stream):
                    experience = tasks.train_stream[i]

                    # eval_streams = [[e] for e in tasks.test_stream[:i + 1]]

                    # res = strategy.train(experiences=experience,
                    #                      eval_streams=eval_streams,
                    #                      pin_memory=pin_memory,
                    #                      num_workers=num_workers)

                    res = strategy.train(experiences=experience,
                                         eval_streams=[
                                             tasks.test_stream[:i + 1]],
                                         pin_memory=pin_memory,
                                         num_workers=num_workers)

                    all_results = strategy.evaluator.get_all_metrics()

                    if eval_every < 0:
                        all_results_dev = strategy.eval(
                            tasks.test_stream[:i + 1],
                            pin_memory=pin_memory,
                            num_workers=num_workers)

                    log.info(res)

                    if save_states:
                        torch.save(strategy,
                                   os.path.join(results_path, f'state_{i}.pt'))

                    # mb_times = strategy.evaluator.all_metric_results['Time_MB/train_phase/train_stream/Task000'][1]
                    # start = 0 if len(mb_time_results) == 0 else sum(map(len, mb_time_results))

                    # mb_time_results.append(mb_times[start:])

                    # train_results.append()
                    results_after_each_task.append(res)
                    # results_after_each_task.append(strategy.eval(tasks.test_stream[:i + 1],
                    #                              pin_memory=pin_memory,
                    #                              num_workers=num_workers))

            with open(results_path, 'w') as f:
                json.dump(results_after_each_task, f, ensure_ascii=False,
                          indent=4)

            with open(complete_results_path, 'w') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

            with open(complete_results_dev_path, 'w') as f:
                json.dump(all_results_dev, f, ensure_ascii=False, indent=4)

        for res in results_after_each_task:
            average_accuracy = []
            for k, v in res.items():
                if k.startswith('Top1_Acc_Stream/eval_phase/test_stream/'):
                    average_accuracy.append(v)
            average_accuracy = np.mean(average_accuracy)
            res['average_accuracy'] = average_accuracy

        # for k, v in train_res[-1].items():
        #     log.info(f'Train Metric {k}:  {v}')

        # for k, v in train_res.items():
        #     if k.startswith('Time_MB') or k.startswith('Time_Epoch'):
        #         log.info(f'Train {k}: {np.mean(v[1]), np.std(v[1])}')
        #
        # for k, v in results_after_each_task[-1].items():
        #     log.info(f'Test Metric {k}:  {v}')
        #
        # all_results.append(results_after_each_task)

    # log.info(f'Average across the experiments.')
    #
    # mean_res = defaultdict(list)
    #
    # for i, r in enumerate(all_results):
    #
    #     for k, v in r[-1].items():
    #         mean_res[k].append(v)
    #
    # m = {k: np.mean(v) for k, v in mean_res.items()}
    # s = {k: np.std(v) for k, v in mean_res.items()}
    #
    # for k, v in results_after_each_task[-1].items():
    #     _m = m[k]
    #     _s = s[k]
    #     log.info(f'Metric {k}: mean: {_m:.2f}, std: {_s:.2f}')


if __name__ == "__main__":
    # try:
    avalanche_training()
    # except Exception as e:
    #     print(e)
