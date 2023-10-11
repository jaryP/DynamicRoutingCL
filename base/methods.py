from avalanche.training import Cumulative, GEM, Replay, Naive, JointTraining, \
    EWC, ICaRL, DER
from avalanche.training.plugins import GEMPlugin, ReplayPlugin

from methods import ContinuosRouting
from methods.strategies import EmbeddingRegularization, \
    ContinualMetricLearning, CustomEWC, SeparatedSoftmaxIncrementalLearning, \
    CoPE, MemoryContinualMetricLearning

from models.utils import AvalanceCombinedModel


def get_plugin(name, **kwargs):
    name = name.lower()
    if name == 'gem':
        return GEMPlugin(
            patterns_per_experience=kwargs['patterns_per_experience'],
            memory_strength=kwargs.get('patterns_per_exp', 0.5))
    elif name == 'none':
        return None
    elif name == 'replay':
        return ReplayPlugin(mem_size=kwargs['mem_size'])
    else:
        assert False


def get_trainer(name, tasks, sit: bool = False, **kwargs):
    name = name.lower()
    num_experiences = len(tasks.train_stream)

    def f(model: AvalanceCombinedModel,
          criterion, optimizer,
          train_epochs: int,
          train_mb_size: int,
          evaluator,
          device,
          eval_every=1, ):

        if name == 'gem':
            return GEM(patterns_per_exp=kwargs['patterns_per_experience'],
                       memory_strength=kwargs.get('memory_strength', 0.5),
                       model=model, criterion=criterion, optimizer=optimizer,
                       train_epochs=train_epochs, train_mb_size=train_mb_size,
                       evaluator=evaluator, device=device,
                       eval_every=eval_every)
        elif name == 'ewc':
            return EWC(ewc_lambda=kwargs['lambda'],
                       mode='separate',
                       keep_importance_data=kwargs.get('keep_importance_data', False),
                       model=model, criterion=criterion,
                       optimizer=optimizer,
                       train_epochs=train_epochs,
                       train_mb_size=train_mb_size,
                       evaluator=evaluator, device=device,
                       eval_every=eval_every)
        elif name == 'oewc':
            return EWC(ewc_lambda=kwargs['lambda'],
                       mode='online',
                       keep_importance_data=kwargs.get('keep_importance_data', False),
                       model=model, criterion=criterion,
                       optimizer=optimizer,
                       train_epochs=train_epochs,
                       train_mb_size=train_mb_size,
                       evaluator=evaluator, device=device,
                       eval_every=eval_every)
        elif name == 'replay':
            return Replay(mem_size=kwargs['mem_size'], model=model,
                          criterion=criterion, optimizer=optimizer,
                          train_epochs=train_epochs,
                          train_mb_size=train_mb_size, evaluator=evaluator,
                          device=device, eval_every=eval_every)
        elif name == 'cumulative':
            return Cumulative(model=model, criterion=criterion,
                              optimizer=optimizer, train_epochs=train_epochs,
                              train_mb_size=train_mb_size, evaluator=evaluator,
                              device=device, eval_every=eval_every)
        elif name == 'naive' or name == 'none':
            return Naive(model=model, criterion=criterion, optimizer=optimizer,
                         train_epochs=train_epochs, train_mb_size=train_mb_size,
                         evaluator=evaluator, device=device,
                         eval_every=eval_every)
        elif name == 'joint':
            return JointTraining(model=model, criterion=criterion,
                                 optimizer=optimizer, train_epochs=train_epochs,
                                 train_mb_size=train_mb_size,
                                 evaluator=evaluator, device=device,
                                 eval_every=eval_every)
        elif name == 'er':
            return EmbeddingRegularization(mem_size=kwargs['mem_size'],
                                           penalty_weight=kwargs.get(
                                               'penalty_weight', 1),
                                           model=model, criterion=criterion,
                                           optimizer=optimizer,
                                           train_epochs=train_epochs,
                                           train_mb_size=train_mb_size,
                                           evaluator=evaluator, device=device,
                                           eval_every=eval_every,
                                           feature_extractor=model.feature_extractor,
                                           classifier=model.classifier)
        elif name == 'cml':
            return ContinualMetricLearning(model=model,
                                           dev_split_size=kwargs.
                                           get('dev_split_size', 100),
                                           penalty_weight=kwargs.
                                           get('penalty_weight', 1),
                                           sit_memory_size=kwargs.
                                           get('sit_memory_size', 500),
                                           proj_w=kwargs.get('proj_w', 1),
                                           merging_strategy=kwargs.get(
                                               'merging_strategy',
                                               'scale_translate'),
                                           memory_parameters=kwargs.get(
                                               'memory_parameters', {}),
                                           memory_type=kwargs.get('memory_type',
                                                                  'random'),
                                           centroids_merging_strategy=kwargs.get(
                                               'centroids_merging_strategy',
                                               'None'),
                                           num_experiences=num_experiences,
                                           optimizer=optimizer,
                                           criterion=criterion,
                                           train_mb_size=train_mb_size,
                                           train_epochs=train_epochs,
                                           device=device, eval_every=eval_every,
                                           sit=sit,
                                           evaluator=evaluator)
        elif name == 'mcml':
            return MemoryContinualMetricLearning(model=model,
                                                 dev_split_size=kwargs.
                                                 get('dev_split_size', 100),
                                                 memory_size=kwargs.
                                                 get('memory_size', 500),
                                                 optimizer=optimizer,
                                                 criterion=criterion,
                                                 train_mb_size=train_mb_size,
                                                 train_epochs=train_epochs,
                                                 device=device,
                                                 eval_every=eval_every,
                                                 sit=sit,
                                                 evaluator=evaluator)
        elif name == 'icarl':
            return ICaRL(feature_extractor=model.feature_extractor,
                         classifier=model.classifier,
                         memory_size=kwargs.get('memory_size'),
                         buffer_transform=kwargs.get('buffer_transform', None),
                         fixed_memory=kwargs.get('fixed_memory', True),
                         optimizer=optimizer,
                         criterion=criterion,
                         train_mb_size=train_mb_size,
                         train_epochs=train_epochs,
                         device=device, eval_every=eval_every,
                         evaluator=evaluator)
        elif name == 'ssil':
            return SeparatedSoftmaxIncrementalLearning(
                model=model,
                sit_memory_size=kwargs.get('memory_size'),
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=train_mb_size,
                train_epochs=train_epochs,
                device=device, eval_every=eval_every,
                evaluator=evaluator
            )
        elif name == 'cope':
            return CoPE(memory_size=kwargs['memory_size'],
                        model=model, criterion=criterion, optimizer=optimizer,
                        train_epochs=train_epochs, train_mb_size=train_mb_size,
                        evaluator=evaluator, device=device,
                        eval_every=eval_every)
        elif name == 'der':
            return DER(model=model, criterion=criterion,
                       mem_size=kwargs['mem_size'],
                       optimizer=optimizer,
                       train_epochs=train_epochs,
                       train_mb_size=train_mb_size,
                       evaluator=evaluator, device=device,
                       eval_every=eval_every,
                       alpha=kwargs['alpha'],
                       beta=kwargs['beta'])
        elif name == 'routing':
            return ContinuosRouting(model=model, criterion=criterion,
                                    optimizer=optimizer,
                                    train_epochs=train_epochs,
                                    train_mb_size=train_mb_size,
                                    evaluator=evaluator, device=device,
                                    eval_every=eval_every,
                                    past_task_reg=kwargs['past_task_reg'],
                                    past_margin=kwargs['past_margin'],
                                    warm_up_epochs=kwargs['warm_up_epochs'],
                                    future_task_reg=kwargs['future_task_reg'],
                                    future_margin=kwargs['future_margin'],
                                    gamma=kwargs['gamma'],
                                    memory_size=kwargs['memory_size'])
        else:
            assert False, f'CL method not found {name.lower()}'

    return f
