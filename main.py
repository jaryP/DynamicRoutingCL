import json
import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
import yaml
from avalanche.benchmarks import data_incremental_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, bwt_metrics, \
    timing_metrics

from avalanche.logging.text_logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin
from omegaconf import DictConfig, OmegaConf
from torch.nn import CrossEntropyLoss

from base.scenario import get_dataset_nc_scenario
from models.base import get_cl_model

# from models.base import get_cl_model

# class CustomTextLogger(StrategyLogger):
#     def __init__(self, file=sys.stdout):
#         super().__init__()
#         self.file = file
#         self.metric_vals = {}
#
#     def log_single_metric(self, name, value, x_plot) -> None:
#         self.metric_vals[name] = (name, x_plot, value)
#
#     def _val_to_str(self, m_val):
#         if isinstance(m_val, torch.Tensor):
#             return '\n' + str(m_val)
#         elif isinstance(m_val, float):
#             return f'{m_val:.4f}'
#         else:
#             return str(m_val)
#
#     def print_current_metrics(self):
#         sorted_vals = sorted(self.metric_vals.values(),
#                              key=lambda x: x[0])
#         for name, x, val in sorted_vals:
#             if isinstance(val, UNSUPPORTED_TYPES):
#                 continue
#             val = self._val_to_str(val)
#             print(f'\t{name} = {val}', file=self.file, flush=True)
#
#     def before_training_exp(self, strategy: 'SupervisedTemplate',
#                             metric_values: List['MetricValue'], **kwargs):
#         super().before_training_exp(strategy, metric_values, **kwargs)
#         self._on_exp_start(strategy)
#
#     def before_eval_exp(self, strategy: 'SupervisedTemplate',
#                         metric_values: List['MetricValue'], **kwargs):
#         super().before_eval_exp(strategy, metric_values, **kwargs)
#         self._on_exp_start(strategy)
#
#     def after_eval_exp(self, strategy: 'SupervisedTemplate',
#                        metric_values: List['MetricValue'], **kwargs):
#         super().after_eval_exp(strategy, metric_values, **kwargs)
#         exp_id = strategy.experience.current_experience
#         task_id = phase_and_task(strategy)[1]
#         if task_id is None:
#             print(f'> Eval on experience {exp_id} '
#                   f'from {stream_type(strategy.experience)} stream ended.',
#                   file=self.file, flush=True)
#         else:
#             print(f'> Eval on experience {exp_id} (Task '
#                   f'{task_id}) '
#                   f'from {stream_type(strategy.experience)} stream ended.',
#                   file=self.file, flush=True)
#         self.print_current_metrics()
#         self.metric_vals = {}
#
#     def before_training(self, strategy: 'SupervisedTemplate',
#                         metric_values: List['MetricValue'], **kwargs):
#         super().before_training(strategy, metric_values, **kwargs)
#         print('-- >> Start of training phase << --', file=self.file, flush=True)
#
#     def before_eval(self, strategy: 'SupervisedTemplate',
#                     metric_values: List['MetricValue'], **kwargs):
#         super().before_eval(strategy, metric_values, **kwargs)
#         print('-- >> Start of eval phase << --', file=self.file, flush=True)
#
#     def after_training(self, strategy: 'SupervisedTemplate',
#                        metric_values: List['MetricValue'], **kwargs):
#         super().after_training(strategy, metric_values, **kwargs)
#         print('-- >> End of training phase << --', file=self.file, flush=True)
#
#     def after_eval(self, strategy: 'SupervisedTemplate',
#                    metric_values: List['MetricValue'], **kwargs):
#         super().after_eval(strategy, metric_values, **kwargs)
#         print('-- >> End of eval phase << --', file=self.file, flush=True)
#         self.print_current_metrics()
#         self.metric_vals = {}
#
#     def _on_exp_start(self, strategy: 'SupervisedTemplate'):
#         action_name = 'training' if strategy.is_training else 'eval'
#         exp_id = strategy.experience.current_experience
#         task_id = phase_and_task(strategy)[1]
#         stream = stream_type(strategy.experience)
#         if task_id is None:
#             print('-- Starting {} on experience {} from {} stream --'
#                   .format(action_name, exp_id, stream),
#                   file=self.file,
#                   flush=True)
#         else:
#             print('-- Starting {} on experience {} (Task {}) from {} stream --'
#                   .format(action_name, exp_id, task_id, stream),
#                   file=self.file,
#                   flush=True)

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
    # model_name = model_cfg['name']

    method = cfg['method']
    plugin_name = method['name'].lower()

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
    eval_every = experiment.get('eval_every', )

    save_states = experiment.get('save_states', False)
    console_log = experiment.get('console_log', False)

    # optimizer_cfg = cfg['optimizer']
    # optimizer_name = optimizer_cfg.get('optimizer', 'sgd')
    # lr = optimizer_cfg.get('lr', 1e-1)
    # momentum = optimizer_cfg.get('momentum', 0.9)
    # weight_decay = optimizer_cfg.get('weight_decay', 0)

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

    force_sit = True if method['name'] == 'der' else False
    force_sit = False

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
            else:
                train_res = results_after_each_task
        else:

            img, _, _ = tasks.train_stream[0].dataset[0]

            if method['name'] == 'der':
                assert 'head_classes' in model_cfg, (
                    'Whn using DER you must specify '
                    'the head dimension of the model, '
                    'by setting head_classes parameter'
                    'in the config file.')

            # cfg1 = {
            #     '_target_': 'hydra.utils.get_class',
            #     'path': 'models.RoutingModel',
            #     # 'layers_dims': [32, 56, 128],
            #     # 'future_paths_to_sample': 2
            # }

            # model = hydra.utils.instantiate(cfg1)
            # m = model([10, 12], 1, 2, 3, 5, 6)

            backbone = hydra.utils.instantiate(cfg.model)

            model = get_cl_model(backbone=backbone,
                                 model_name='',
                                 input_shape=tuple(img.shape),
                                 method_name=plugin_name,
                                 is_class_incremental_learning=is_cil,
                                 **model_cfg)

            eval_plugin = EvaluationPlugin(
                accuracy_metrics(stream=True,
                                 trained_experience=True, experience=True),
                bwt_metrics(experience=True, stream=True),
                # timing_metrics(minibatch=True, epoch=True, experience=False),
                # loggers=[TextLogger()] if console_log else [],
                loggers=[TextLogger()],
            )

            opt = hydra.utils.instantiate(cfg.optimizer,
                                          params=model.parameters())

            # opt = get_optimizer(parameters=model.parameters(),
            #                     name=optimizer_name,
            #                     lr=lr,
            #                     weight_decay=weight_decay,
            #                     momentum=momentum)

            criterion = CrossEntropyLoss()

            if cfg is not None and 'method' in cfg.method:
                if method.name == 'ICaRL':
                    strategy = hydra.utils.instantiate(cfg.method.method,
                                                       feature_extractor=model.feature_extractor,
                                                       classifier=model.classifier,
                                                       optimizer=opt,
                                                       train_epochs=epochs,
                                                       train_mb_size=batch_size,
                                                       evaluator=eval_plugin,
                                                       device=device,
                                                       eval_every=eval_every)
                else:
                    strategy = hydra.utils.instantiate(cfg.method.method,
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

            indexes = np.arange(len(tasks.train_stream))

            if method['name'].lower() == 'cope':
                tasks = data_incremental_benchmark(tasks, batch_size,
                                                   shuffle=True)
                indexes = np.arange(len(tasks.train_stream))

                if method.get('shuffle', False):
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

                    eval_streams = [[e] for e in tasks.test_stream[:i + 1]]

                    res = strategy.train(experiences=experience,
                                         eval_streams=eval_streams,
                                         pin_memory=pin_memory,
                                         num_workers=num_workers)

                    # train_res = strategy.evaluator.all_metric_results

                    all_results = strategy.evaluator.get_all_metrics()
                    print(all_results, res)

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
