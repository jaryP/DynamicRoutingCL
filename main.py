import json
import logging
import os
import random
import warnings
from collections import defaultdict
from types import MethodType
from typing import Sequence

import hydra
import numpy as np
import torch
from avalanche.benchmarks import data_incremental_benchmark
from avalanche.logging import WandBLogger, TextLogger

from avalanche.training import DER
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import methods.routing
from base.scenario import get_dataset_nc_scenario
from models.base import get_cl_model


def make_train_dataloader(
    self,
    num_workers=0,
    shuffle=True,
    pin_memory=None,
    persistent_workers=False,
    drop_last=False,
    **kwargs
):
    """Data loader initialization.

    Called at the start of each learning experience after the dataset
    adaptation.

    :param num_workers: number of thread workers for the data loading.
    :param shuffle: True if the data should be shuffled, False otherwise.
    :param pin_memory: If True, the data loader will copy Tensors into CUDA
        pinned memory before returning them. Defaults to True.
    """

    assert self.adapted_dataset is not None

    torch.utils.data.DataLoader

    other_dataloader_args = self._obtain_common_dataloader_parameters(
        batch_size=self.train_mb_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    self.dataloader = DataLoader(
        self.adapted_dataset, **other_dataloader_args
    )


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

    device = cfg.get('device', 'cpu')

    scenario = cfg['scenario']
    dataset = scenario['dataset']
    n_tasks = scenario['n_tasks']
    task_incremental_learning = scenario['return_task_id']
    permuted_dataset = scenario.get('permuted_dataset', False)

    shuffle = scenario['shuffle']
    shuffle_first = scenario.get('shuffle_first', True)

    model_cfg = cfg['model']

    plugin_name = cfg.trainer_name.lower()

    training = cfg['training']
    epochs = training['epochs']
    batch_size = training['batch_size']

    num_workers = training.get('num_workers', 0)
    pin_memory = training.get('pin_memory', False)
    use_standard_dataloader = training.get('use_standard_dataloader', True)
    dev_split = training.get('dev_split', None)

    experiment = cfg['experiment']
    n_experiments = experiment.get('experiments', 1)
    load = experiment.get('load', True)

    save = experiment.get('save', True)
    plot = experiment.get('plot', False)
    eval_every = experiment.get('eval_every', 5)

    save_states = experiment.get('save_states', False)

    if device == 'cpu':
        warnings.warn("Device set to cpu.")
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn(f"Device not found {device} "
                      f"or CUDA {torch.cuda.is_available()}")

    device = torch.device(device)

    base_path = os.getcwd()
    cfg_path = os.path.join(os.getcwd(), '.hydra', 'config.yaml')

    is_cil = not task_incremental_learning
    task_incremental_learning = task_incremental_learning if plugin_name != 'cml' \
        else True

    force_sit = True if plugin_name == 'der' else False
    force_sit = False

    head_classes = cfg.model.get('head_classes', None)
    if plugin_name == 'der':
        assert head_classes is not None, (
            'When using DER you must specify '
            'the head dimension of the model, '
            'by setting head_classes parameter'
            'in the config file.')

        del cfg.model.head_classes

    experiments_results = []
    tasks_split_dict = {}

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
                                        shuffle=shuffle,
                                        seed=seed, force_sit=force_sit,
                                        method_name=plugin_name,
                                        permuted_dataset=permuted_dataset,
                                        dev_split=dev_split)

        log.info(f'Original classes: {tasks.classes_order_original_ids}')
        log.info(f'Original classes per exp: {tasks.original_classes_in_exp}')

        tasks_split_dict[seed] = {
            'original_classes': tasks.classes_order_original_ids,
            'tasks_classes': [list(v) for v in tasks.original_classes_in_exp]}

        with open(os.path.join(base_path, 'tasks_split.json'), 'w') as f:
            json.dump(tasks_split_dict, f, ensure_ascii=False, indent=4)

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
                    experiments_results.append(last_results)
            else:
                train_res = results_after_each_task
        else:

            img, _, _ = tasks.train_stream[0].dataset[0]

            backbone = hydra.utils.instantiate(cfg.model)

            model = get_cl_model(backbone=backbone,
                                 model_name='',
                                 input_shape=tuple(img.shape),
                                 method_name=plugin_name,
                                 head_classes=head_classes,
                                 is_class_incremental_learning=is_cil,
                                 **model_cfg)

            metrics = hydra.utils.instantiate(cfg.evaluation.metrics)

            loggers = []

            if cfg.evaluation.get('enable_textlog', True):
                loggers.append(TextLogger())

            if cfg.evaluation.get('enable_wandb', True):
                wandb_group = cfg.get('wandb_group', None)
                wandb_prefix = cfg.get('wandb_prefix', '')

                if dev_split is not None:
                    tags = ['grid_search']
                else:
                    tags = ['train']

                _tags = cfg.get('wadnb_tags', [])
                print(_tags, type(_tags))
                if not isinstance(_tags, ListConfig):
                    _tags = [_tags]
                print(_tags)
                tags += _tags

                wandb_name = f'{cfg.scenario.dataset}/{cfg.scenario.n_tasks}_{cfg.trainer_name}_{backbone.__class__.__name__}_{exp_n}'

                if permuted_dataset:
                    wandb_name = 'RP_' + wandb_name

                if wandb_prefix is not None and wandb_prefix != '':
                    wandb_name = wandb_prefix + wandb_name

                wandb_dict = OmegaConf.to_container(cfg, resolve=True)
                wandb_dict['saving_path'] = experiment_path

                v = WandBLogger(project_name=cfg.core.project_name,
                                run_name=wandb_name,
                                params={'config': wandb_dict,
                                        'reinit': True,
                                        'group': wandb_group,
                                        'tags': tags})

            # for ev in cfg.evaluation.loggers:
            #     if 'WandBLogger' in ev['_target_']:
            #         wandb_group = cfg.get('wandb_group', None)
            #         wandb_prefix = cfg.get('wandb_prefix', '')
            #
            #         if dev_split is not None:
            #             tags = ['grid_search']
            #         else:
            #             tags = ['train']
            #
            #         _tags = cfg.get('wadnb_tags', [])
            #         if not isinstance(_tags, Sequence):
            #             _tags = [_tags]
            #         tags += _tags
            #
            #         wandb_name = f'{cfg.scenario.dataset}/{cfg.scenario.n_tasks}_{cfg.trainer_name}_{backbone.__class__.__name__}_{exp_n}'
            #
            #         if permuted_dataset:
            #             wandb_name = 'RP_' + wandb_name
            #
            #         if wandb_prefix is not None and wandb_prefix != '':
            #             wandb_name = wandb_prefix + wandb_name
            #
            #         wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            #         wandb_dict['saving_path'] = experiment_path
            #
            #         v = WandBLogger(project_name=cfg.core.project_name,
            #                         run_name=wandb_name,
            #                         params={'config': wandb_dict,
            #                                 'reinit': True,
            #                                 'group': wandb_group,
            #                                 'tags': tags})
            #     else:
            #         v = hydra.utils.instantiate(ev)
                loggers.append(v)

            eval_plugin = EvaluationPlugin(
                metrics,
                loggers=loggers,
                strict_checks=False
            )

            opt = hydra.utils.instantiate(cfg.optimizer,
                                          params=model.parameters())

            criterion = CrossEntropyLoss()

            method_name = hydra.utils.get_class(cfg.method._target_).__name__.lower()

            if cfg is not None and '_target_' in cfg.method:
                if plugin_name == 'icarl':
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
                    if method_name == 'podnet':
                        strategy = hydra.utils.instantiate(cfg.method,
                                                           feature_extractor=model.feature_extractor,
                                                           classifier=model.classifier,
                                                           criterion=criterion,
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

            if use_standard_dataloader:
                strategy.make_train_dataloader = MethodType(make_train_dataloader, strategy)

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

            experiments_results.append(results_after_each_task)

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

    mean_res = defaultdict(list)

    for i, r in enumerate(experiments_results):
        for k, v in r[-1].items():
            mean_res[k].append(v)

    averaged_results = {k: {'mean': np.mean(v),
                            'std': np.std(v)} for k, v in mean_res.items()}

    for k, d in averaged_results.items():
        log.info(f'Metric {k}: mean: {d["mean"]:.2f}, std: {d["std"]:.2f}')

    with open(os.path.join(base_path, 'averaged_results.json'), 'w') as f:
        json.dump(averaged_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # try:
    avalanche_training()
    # except Exception as e:
    #     print(e)
