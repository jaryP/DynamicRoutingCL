from typing import Tuple, Dict

import torch
from avalanche.benchmarks import CLExperience
from avalanche.benchmarks.utils import AvalancheDataset, ConstantSequence
from avalanche.models import MultiTaskModule, DynamicModule
from torch import nn


class CustomMultiHeadClassifier(MultiTaskModule):
    def __init__(self, in_features, heads_generator, out_features=None,
                 p=None):

        super().__init__()

        self.heads_generator = heads_generator
        self.in_features = in_features
        self.starting_out_features = out_features
        self.classifiers = torch.nn.ModuleDict()

    def adaptation(self, dataset: AvalancheDataset):
        super().adaptation(dataset)
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)  # need str keys
            if tid not in self.classifiers:

                if self.starting_out_features is None:
                    out = max(dataset.targets) + 1
                else:
                    out = self.starting_out_features

                new_head = self.heads_generator(self.in_features, out)
                self.classifiers[tid] = new_head

    def forward_single_task(self, x, task_label, **kwargs):
        return self.classifiers[str(task_label)](x, **kwargs)


class AvalanceCombinedModel(MultiTaskModule):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, p=None):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

        # self.dropout = lambda x: x
        # if p is not None:
        #     self.dropout = Dropout(p)

    def forward_single_task(self, x: torch.Tensor, task_label: int,
                            return_embeddings: bool = False,
                            t=None):

        out = self.feature_extractor(x, task_labels=task_label)
        out = torch.flatten(out, 1)

        # out = self.dropout(out)

        logits = self.classifier(out, task_labels=task_label)

        if return_embeddings:
            return out, logits

        return logits

    def forward_all_tasks(self, x: torch.Tensor,
                          return_embeddings: bool = False,
                          **kwargs):

        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x,
                                                    task_id,
                                                    return_embeddings,
                                                    **kwargs)
        return res

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor,
                return_embeddings: bool = False,
                **kwargs) \
            -> torch.Tensor:

        if task_labels is None:
            return self.forward_all_tasks(x,
                                          return_embeddings=return_embeddings,
                                          **kwargs)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, return_embeddings,
                                            **kwargs)

        unique_tasks = torch.unique(task_labels)
        if len(unique_tasks) == 1:
            return self.forward_single_task(x, unique_tasks.item(),
                                            return_embeddings, **kwargs)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item(),
                                                return_embeddings, **kwargs)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out


class PytorchCombinedModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, p=None):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


class FxCombinedModel(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 classifier: nn.Module, p=None):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor, **kwargs):
        d = self.feature_extractor(x)
        logits = self.classifier(d['features'])

        return logits, d


class LocalSimilarityClassifier(DynamicModule):
    def __init__(self, in_features, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._in_features = in_features
        self._vectors = nn.ParameterList()
        # self._vectors = nn.Parameter(torch.randn((1, 100,  self._in_features)))

        self._scaler = nn.Parameter(torch.ones([1]))

    # @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        device = self._adaptation_device
        curr_classes = experience.classes_seen_so_far

        old_nclasses = len(self._vectors)
        new_nclasses = len(curr_classes)

        if old_nclasses == new_nclasses:
            return

        # print([v[:10] for v in self._vectors])
        for _ in range(new_nclasses - old_nclasses):
            v = torch.randn((1, self._in_features),
                            device=device)
            v = nn.Parameter(v, requires_grad=True)

            self._vectors.append(v)

        # old_w = self._vectors
        #
        # self._vectors = nn.Parameter(torch.randn((1, new_nclasses,
        #                                           self._in_features),
        #                                          device=device),
        #                              requires_grad=True)
        # self._vectors[0, :old_nclasses] = old_w

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        ls = []
        for v in self._vectors:
            l = nn.functional.cosine_similarity(x, v, -1)
            ls.append(l)

        ls = torch.cat(ls, -1)
        # v = self._vectors.unsqueeze(0)
        # v = self._vectors
        # ls = nn.functional.cosine_similarity(x, v, -1)

        return ls * self._scaler


class bn_track_stats:
    def __init__(self, module: nn.Module, condition=True):
        self.module = module
        self.enable = condition

    def __enter__(self):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False

    def __exit__(self, type, value, traceback):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True
