from copy import deepcopy

import numpy as np
import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader


class LogitsDataset:
    def __init__(self, base_datasets, logits, features=None, classes=None):
        self.base_dataset = base_datasets
        self.logits = logits
        self.features = features
        self.current_classes = classes

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, item):
        if self.features is None:
            return *self.base_dataset[item], self.logits[item]

        return *self.base_dataset[item], self.logits[item], self.features[item]

    def train(self):
        self.base_dataset = self.base_dataset.train()
        return self

    def eval(self):
        self.base_dataset = self.base_dataset.eval()
        return self

    def subset(self, indexes):
        self.base_dataset = self.base_dataset.subset(indexes)
        self.logits = self.logits[indexes]

        if self.features is not None:
            self.features = self.features[indexes]

        return self


class Margin(SupervisedTemplate):
    def __init__(self, model,
                 optimizer: Optimizer,
                 mem_size: int,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 alpha: float = 0,
                 past_task_reg=1,
                 rehearsal_metric='mse',
                 margin_type='adaptive',
                 past_margin=0.5,
                 warm_up_epochs=-1,
                 future_task_reg=1,
                 future_margin=3,
                 gamma=1,
                 use_logits_memory=False,
                 cumulative_memory=True,
                 reg_sampling_bs=None,
                 eval_mb_size: int = None,
                 device=None,
                 plugins=None,
                 batch_size_mem=None,
                 criterion=None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 ):

        assert margin_type in ['fixed', 'adaptive', 'normal', 'mean']
        assert 0 <= past_margin <= 1

        assert rehearsal_metric in ['kl', 'mse']

        self.future_margins = []
        self.tasks_nclasses = dict()
        self.current_mb_size = 0
        self.future_logits = None

        self.cumulative_memory = cumulative_memory

        self.double_sampling = -1 if reg_sampling_bs is None else reg_sampling_bs

        self.past_task_reg = past_task_reg
        self.past_margin = past_margin
        self.margin_type = margin_type
        self.warm_up_epochs = warm_up_epochs

        self.future_margin = future_margin
        self.future_task_reg = future_task_reg

        self.alpha = alpha
        # Past task logits weight
        self.gamma = gamma

        self.logit_regularization = rehearsal_metric
        self.tau = 1

        self.memory_size = mem_size

        self.use_logits_memory = use_logits_memory

        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem

        self.is_finetuning = False
        self.mb_future_features = None

        self.past_dataset = {}
        self.past_dataset_tasks = {}
        self.past_model = None

        super().__init__(
            model=model, optimizer=optimizer, criterion=None,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_forward(self, **kwargs):
        self.mb_output, self.future_logits = self.mb_output
        super()._after_forward(**kwargs)

    def _after_eval_forward(self, **kwargs):
        self.mb_output, self.future_logits = self.mb_output
        self.mb_output = torch.cat(self.mb_output, -1)

    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        if self.use_logits_memory:
            self._after_training_exp_logits()
        else:
            self._after_training_exp_f(**kwargs)

        super()._after_training_exp(**kwargs)

        classes_so_far = len(self.experience.classes_seen_so_far)

    def sample_past_batch(self, batch_size, strict=True):
        classes = self.past_dataset
        ln = len(classes)

        if ln <= 0:
            return None

        samples_per_task = max(batch_size // ln, 1)
        rest = batch_size % ln

        if rest > 0:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        x, y, t = [], [], []
        ls = []

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            batch = next(iter(DataLoader(d,
                                                      batch_size=bs,
                                                      shuffle=True)))

            if len(batch) == 3:
                ot_x, ot_y, ot_tid = batch
            else:
                ot_x, ot_y, ot_tid, l = batch
                ls.append(l)

            ot_tid = torch.full_like(ot_tid, self.past_dataset_tasks[c])
            x.append(ot_x)
            y.append(ot_y)
            t.append(ot_tid)

        if len(ls) > 0:
            ls = torch.cat(ls)
        return torch.cat(x, 0), torch.cat(y, 0), torch.cat(t, 0), ls

    def _before_forward_f(self, **kwargs):

        super()._before_forward(**kwargs)

        if len(self.past_dataset) == 0:
            return None

        batch = self.sample_past_batch(len(self.mb_x))
        if len(batch) == 3:
            x, y, t = batch
        else:
            x, y, t, l = batch
            self.past_logits = l

        self.current_mb_size = len(self.mb_x)

        current_task = self.experience.current_experience
        self.mbatch[2] = torch.full_like(self.mbatch[2], current_task)

        self.mbatch[0] = torch.cat((self.mbatch[0], x.to(self.device)))
        self.mbatch[1] = torch.cat((self.mbatch[1], y.to(self.device)))
        self.mbatch[2] = torch.cat((self.mbatch[2], t.to(self.device)))

    @torch.no_grad()
    def _after_training_exp_f(self, **kwargs):

        if self.cumulative_memory:
            samples_to_save = self.memory_size // len(
                self.experience.classes_seen_so_far)
        else:
            samples_to_save = self.memory_size

        tid = self.experience.current_experience
        
        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)
                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)

            self.past_dataset[y] = dataset.train().subset(indexes)
            self.past_dataset_tasks[y] = tid

        # if self.experience.current_experience > 0:
        self.past_model = deepcopy(self.model)
        self.past_model.eval()

    @torch.no_grad()
    def _after_training_exp_logits(self, **kwargs):
        tid = self.experience.current_experience

        samples_to_save = self.memory_size // len(
            self.experience.classes_seen_so_far)

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)

                # all_logits = []
                # for x, y, _, l in DataLoader(d, batch_size=self.train_mb_size):
                #     _l = self.model(x.to(self.device))[0]
                #     _l = _l.cpu()
                #     l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
                #     all_logits.append(l)
                #
                # all_logits = torch.cat(all_logits, 0)
                # d.logits = all_logits

                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        self.model.eval()
        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)
            all_logits = []
            for x, _, _ in DataLoader(dataset.eval().subset(indexes),
                                      batch_size=self.eval_mb_size):
                x = x.to(self.device)

                logits = self.model(x)[0]
                logits = torch.cat(logits, -1)
                all_logits.append(logits.cpu())

            all_logits = torch.cat(all_logits, 0)

            d = LogitsDataset(dataset.train().subset(indexes), all_logits)

            self.past_dataset[y] = d
            self.past_dataset_tasks[y] = tid

        # if self.experience.current_experience > 0:
        self.past_model = deepcopy(self.model)
        self.past_model.eval()

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        self._before_forward_f(**kwargs)

    @torch.no_grad()
    def _before_training_exp(self, **kwargs):
        tid = self.experience.current_experience

        super()._before_training_exp(**kwargs)

        if len(self.past_dataset) > 0 and not self.use_logits_memory:
            self.past_model = deepcopy(self.model)
            self.past_model.eval()

        self.future_margins.append(self.future_margin)

        if self.use_logits_memory:
            self.model.eval()

            for k, d in self.past_dataset.items():
                all_logits = []
                for x, y, _, l in DataLoader(d.eval(), batch_size=self.train_mb_size):
                    _l = self.model(x.to(self.device))[0]
                    _l = _l[tid].cpu()
                    l = torch.cat((l, _l), -1)
                    all_logits.append(l)

                all_logits = torch.cat(all_logits, 0)
                d.logits = all_logits

                self.past_dataset[k] = d

            self.model.train()

        # self.future_margin =      zself.future_margin + self.past_margin * 2
        # self.future_margins.append(self.future_margin)

    def _before_training_epoch(self, **kwargs):
        tid = self.experience.current_experience
        if tid == 0:
            return

        with torch.no_grad(), torch.inference_mode():
            logits = []

            for x, y, t in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)

                preds, _ = self.model(x)
                past = torch.cat(preds, -1)

                preds = torch.cat(preds, -1)
                preds = torch.softmax(preds, -1)

                # cl = preds[:, :past.shape[-1]].max(-1).values

                cl = preds[range(len(preds)), y]

                logits.append(cl)

            # logits = torch.cat(logits, 0) / 2
            logits = torch.cat(logits, 0)

            if self.margin_type == 'mean':
                self._margin = 1 - logits.mean()
            elif self.margin_type != 'normal':
                self.margin_distribution = torch.distributions.normal.Normal(logits.mean(), logits.std())

    def criterion(self):
        tid = self.experience.current_experience

        if not self.is_training:
            loss_val = nn.functional.cross_entropy(self.mb_output,
                                                   self.mb_y)

            return loss_val

        if len(self.past_dataset) == 0:
            pred = torch.cat(self.mb_output, -1)
            # if self.future_logits is not None:
            #     pred = torch.cat((pred, self.future_logits), -1)
            loss = nn.functional.cross_entropy(pred, self.mb_y,
                                               label_smoothing=0)
        else:
            # TODO: REMOVE
            # if self.clock.train_exp_iterations == 0:
            #     self.get_gradients()

            ce_loss = 0
            margin_loss = 0
            margin_den = 0
            ce_den = 0

            bs = len(self.mb_x)

            for t in torch.unique(self.mb_task_id):
                if t < tid and self.alpha <= 0:
                    continue
                    # loss = loss * self.alpha

                mask = self.mb_task_id == t
                ce_den += mask.sum().item()

                y = self.mb_y[mask]
                co = torch.cat(self.mb_output[t.item():], -1)[mask]

                if t > 0:
                    past = torch.cat(self.mb_output[:t.item()], -1)[mask]

                    distr = torch.cat((past, co), -1)
                    distr = torch.softmax(distr, -1)

                    mx_current_classes = distr[range(len(co)), y]
                    past_max = distr[:, :past.shape[-1]].max(-1).values

                    y = y - past.shape[-1]

                    # if 10 > 0 and self.clock.train_exp_epochs / 10 >= 1:
                        # past_max = past.max(-1).values.detach()
                        # past_max = past.max(-1).values
                        #
                        # mx_current_classes = co[range(len(co)), y]
                        # margin_dist = torch.relu(past_max - mx_current_classes + self.past_margin)
                        # margin = (1 / (distr.shape[-1] - 1))

                    if self.margin_type == 'fixed':
                        margin = self.past_margin
                    elif self.margin_type == 'normal':
                        margin = self.margin_distribution.rsample([len(distr)])
                        margin = torch.clip(margin, 0, 1)
                        margin = margin * self.past_margin
                        margin = torch.minimum(margin, torch.full_like(margin, (1 / (distr.shape[-1] - 1))))
                    elif self.margin_type == 'mean':
                        margin = (1 - mx_current_classes.mean()) * self.past_margin
                        # margin = mx_current_classes.mean() * self.past_margin
                        # margin = torch.minimum(margin,
                        #                        torch.full_like(margin, (1 / (distr.shape[-1]))))
                    else:
                        margin = (1 / (distr.shape[-1] - 1))

                    margin_dist = torch.relu(
                        past_max - mx_current_classes + margin)

                    den_mask = margin_dist > 0

                    margin_loss += margin_dist.mean()

                    # if den_mask.sum() > 0:
                    #     margin_den += den_mask.sum()
                    #     past_reg = margin_dist[den_mask].sum()
                    #
                    #     margin_loss += past_reg

                # if self.future_logits is not None:
                #     co = torch.cat((co, self.future_logits), -1)

                loss = nn.functional.cross_entropy(co, y, reduction='sum')

                if t < tid:
                    loss = loss * self.alpha

                ce_loss += loss

            # if margin_den > 0:
            #     margin_loss = (margin_loss / margin_den) * self.past_task_reg

            w = 1
            if self.warm_up_epochs > 0:
                w = min(1, (self.clock.train_exp_epochs / self.warm_up_epochs) ** 2)

            margin_loss = margin_loss * self.past_task_reg * w
            ce_loss = (ce_loss / ce_den)

            loss = ce_loss + margin_loss

            if self.gamma > 0:
                factor = 1
                # if self.scheduled_factor:
                #     current_classes = len(
                #         self.experience.classes_in_this_experience)
                #     seen_classes = len(self.experience.classes_seen_so_far)
                #
                #     factor = (seen_classes / current_classes) ** 0.5
                # else:
                #     factor = 1

                if len(self.past_dataset) > 0:
                    x, y, tids = self.mbatch
                    curr_logits = self.mb_output
                    mask = tids != tid

                    if self.use_logits_memory:
                        past_logits = self.past_logits.to(x.device)
                    else:
                        with torch.no_grad():
                            past_logits, _ = self.past_model(x)
                            past_logits = torch.cat(past_logits, -1)[mask]

                    co = torch.cat(curr_logits, -1)[mask]

                    if self.logit_regularization == 'mse':
                        past_reg_loss = nn.functional.mse_loss(co, past_logits,
                                                               reduction='mean')
                    else:
                        past_reg_loss = nn.functional.kl_div(
                            torch.log_softmax(co, -1),
                            torch.softmax(past_logits, -1),
                            reduction='batchmean')

                    loss = loss + past_reg_loss * self.gamma * factor

        future_reg = 0
        if self.future_task_reg > 0 and self.future_margin > 0 and self.future_logits is not None:
            future_logits = self.future_logits

            future_mx = future_logits.min(-1).values.detach()
            current_mx = torch.cat(self.mb_output, -1).max(-1).values

            margin = self.future_margin

            reg = current_mx - future_mx - margin
            logits_pos_reg = - current_mx + future_mx

            reg = reg + logits_pos_reg
            reg = torch.relu(reg)
            mask = reg > 0
            if any(mask):
                future_reg = reg[mask].mean()

            future_reg = future_reg * self.future_task_reg

            loss += future_reg

        return loss


class OldMargin(SupervisedTemplate):
    def __init__(self, model,
                 optimizer: Optimizer,
                 mem_size: int,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 alpha: float = 1,
                 past_task_reg=1,
                 past_margin=0.2,
                 warm_up_epochs=-1,
                 future_task_reg=1,
                 future_margin=3,
                 gamma=1,
                 reg_sampling_bs=None,
                 eval_mb_size: int = None,
                 device=None,
                 plugins=None,
                 batch_size_mem=None,
                 criterion=None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 ):

        self.future_margins = []
        self.tasks_nclasses = dict()
        self.current_mb_size = 0

        self.double_sampling = -1 if reg_sampling_bs is None else reg_sampling_bs

        # Past task loss weight
        self.alpha = alpha

        self.past_task_reg = past_task_reg
        self.past_margin = past_margin

        self.future_task_reg = future_task_reg
        self.future_margin = future_margin

        # Past task logits weight
        self.gamma = gamma
        self.logit_regularization = 'kl'
        self.tau = 1

        self.memory_size = mem_size

        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem

        self.is_finetuning = False
        self.mb_future_features = None

        self.past_dataset = {}
        self.past_model = None

        super().__init__(
            model=model, optimizer=optimizer, criterion=None,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_eval_forward(self, **kwargs):
        self.mb_output = torch.cat(self.mb_output, -1)

    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        self._after_training_exp_f(**kwargs)

        super()._after_training_exp(**kwargs)

        classes_so_far = len(self.experience.classes_seen_so_far)

    def sample_past_batch(self, batch_size, strict=True):
        classes = self.past_dataset
        ln = len(classes)

        if ln <= 0:
            return None

        samples_per_task = max(batch_size // ln, 1)
        rest = batch_size % ln

        if rest > 0:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        x, y, t = [], [], []

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            ot_x, ot_y, ot_tid = next(iter(DataLoader(d,
                                                      batch_size=bs,
                                                      shuffle=True)))

            x.append(ot_x)
            y.append(ot_y)
            t.append(ot_tid)

        return torch.cat(x, 0), torch.cat(y, 0), torch.cat(t, 0)

    def _before_forward_f(self, **kwargs):

        super()._before_forward(**kwargs)

        if len(self.past_dataset) == 0:
            return None

        x, y, t = self.sample_past_batch(len(self.mb_x))
        self.current_mb_size = len(self.mb_x)

        self.mbatch[0] = torch.cat((self.mbatch[0], x.to(self.device)))
        self.mbatch[1] = torch.cat((self.mbatch[1], y.to(self.device)))
        self.mbatch[2] = torch.cat((self.mbatch[2], t.to(self.device)))

    @torch.no_grad()
    def _after_training_exp_f(self, **kwargs):

        samples_to_save = self.memory_size // len(
            self.experience.classes_seen_so_far)

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)
                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)

            self.past_dataset[y] = dataset.train().subset(indexes)

        # if self.experience.current_experience > 0:
        self.past_model = deepcopy(self.model)
        self.past_model.eval()

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        self._before_forward_f(**kwargs)

    @torch.no_grad()
    def _before_training_exp(self, **kwargs):

        super()._before_training_exp(**kwargs)

        if len(self.past_dataset) > 0:
            self.past_model = deepcopy(self.model)
            self.past_model.eval()

        self.future_margins.append(self.future_margin)
        # self.future_margin = self.future_margin + self.past_margin * 2
        # self.future_margins.append(self.future_margin)

    def criterion(self):
        tid = self.experience.current_experience

        if not self.is_training:
            loss_val = nn.functional.cross_entropy(self.mb_output,
                                                   self.mb_y)

            return loss_val

        past_reg = 0
        future_reg = 0

        if len(self.past_dataset) == 0:
            pred = torch.cat(self.mb_output, -1)
            loss = nn.functional.cross_entropy(pred, self.mb_y,
                                               label_smoothing=0)
        else:
            ce_loss = 0
            margin_loss = 0
            margin_den = 0
            ce_den = 0

            bs = len(self.mb_x)

            for t in torch.unique(self.mb_task_id):
                if t < tid and self.alpha <= 0:
                    continue
                    # loss = loss * self.alpha

                mask = self.mb_task_id == t
                ce_den += mask.sum().item()

                y = self.mb_y[mask]
                co = torch.cat(self.mb_output[t.item():], -1)[mask]

                if t > 0:
                    past = torch.cat(self.mb_output[:t.item()], -1)[mask]

                    distr = torch.cat((past, co), -1)
                    distr = torch.softmax(distr, -1)

                    mx_current_classes = distr[range(len(co)), y]
                    past_max = distr[:, :past.shape[-1]].max(-1).values

                    y = y - past.shape[-1]

                    # past_max = past.max(-1).values.detach()
                    # past_max = past.max(-1).values
                    #
                    # mx_current_classes = co[range(len(co)), y]

                    # margin_dist = torch.relu(past_max - mx_current_classes + self.past_margin)
                    # margin_dist = torch.relu(past_max - mx_current_classes + (1 / (distr.shape[-1] - 1)))
                    margin = mx_current_classes.mean() / 2
                    margin_dist = torch.relu(
                        past_max - mx_current_classes + margin)

                    den_mask = margin_dist > 0
                    margin_den += den_mask.sum().item()

                    past_reg = margin_dist[den_mask].sum()

                    margin_loss += past_reg

                loss = nn.functional.cross_entropy(co, y, reduction='sum')

                if t < tid:
                    loss = loss * self.alpha

                ce_loss += loss

            margin_loss = (margin_loss / margin_den) * self.past_task_reg
            ce_loss = (ce_loss / ce_den)

            loss = ce_loss + margin_loss

            if any([self.gamma > 0, self.delta > 0]):
                if self.scheduled_factor:
                    current_classes = len(
                        self.experience.classes_in_this_experience)
                    seen_classes = len(self.experience.classes_seen_so_far)

                    factor = (seen_classes / current_classes) ** 0.5
                else:
                    factor = 1

                if self.past_model is not None:
                    bs = len(self.mb_output) // 2

                    x, y, tids = self.mbatch
                    # if self.double_sampling > 0:
                    # x, y, tid = self.sample_past_batch(self.double_sampling)
                    # x, y = x.to(self.device), y.to(self.device)
                    # curr_logits = self.model(x)[0]

                    curr_logits = self.mb_output

                    # curr_logits = torch.cat((curr_logits, random_logits), -1)
                    # else:
                    #     x, y = (self.mb_x[self.current_mb_size:],
                    #             self.mb_y[self.current_mb_size:])
                    #     # curr_logits = self.mb_output[ self.current_mb_size:]
                    #     curr_features = self.mb_features[self.current_mb_size:]
                    #
                    #     curr_logits = pred[self.current_mb_size:]
                    #     # curr_features = self.mb_features[ self.current_mb_size:]

                    with torch.no_grad():
                        # past_logits, past_features, future_logits, _ = self.past_model(x, other_paths=self.model.current_random_paths)
                        past_logits = self.past_model(x)
                        # curr_logits = curr_logits[:, :past_logits.shape[1]]
                    # if not self.double_sampling > 0:
                    #     curr_logits = curr_logits[:, :past_logits.shape[-1]]

                    past_reg_loss = 0

                    if self.alpha <= 0:
                        mask = tids != tid

                        co = torch.cat(curr_logits, -1)[mask]
                        po = torch.cat(past_logits, -1)[mask]
                        lr = nn.functional.mse_loss(co, po, reduction='mean')
                        # past_reg_loss = nn.functional.kl_div(torch.log_softmax(co, -1),
                        #                           torch.softmax(po, -1),
                        #                           reduction='batchmean')
                        # a = 1 / mask.sum()

                        # for pl, cl in zip(past_logits, curr_logits):
                        #     pl = pl[mask]
                        #     cl = cl[mask]
                        #
                        #     # cl = nn.functional.normalize(cl, 2, -1)
                        #     # pl = nn.functional.normalize(pl, 2, -1)
                        #     # lr = nn.functional.kl_div(torch.log_softmax(cl, -1),
                        #     #                           torch.softmax(pl, -1),
                        #     #                           reduction='batchmean')
                        #     lr = nn.functional.mse_loss(cl, pl, reduction='mean')
                        past_reg_loss += lr
                    else:
                        for t in torch.unique(tids):
                            if t == tid:
                                continue

                            mask = tids == t

                            _y = y[mask]
                            co = torch.cat(curr_logits[t.item():], -1)[mask]
                            po = torch.cat(past_logits[t.item():], -1)[mask]

                            lr = nn.functional.mse_loss(co, po, reduction='sum')
                            # lr = nn.functional.kl_div(torch.log_softmax(co, -1),
                            #                           torch.softmax(po, -1), reduction='sum')
                            past_reg_loss += lr

                        past_reg_loss = past_reg_loss / (len(x) / 2)

                    loss = loss + past_reg_loss * self.gamma * factor

                    # if self.gamma > 0:
                    #     if self.logit_regularization == 'kl':
                    #         curr_logits = torch.log_softmax(
                    #             curr_logits / self.tau, -1)
                    #         past_logits = torch.log_softmax(
                    #             past_logits / self.tau, -1)
                    #
                    #         lr = nn.functional.kl_div(curr_logits, past_logits,
                    #                                   log_target=True,
                    #                                   reduction='batchmean')
                    #     elif self.logit_regularization == 'mse':
                    #         classes = len(self.past_dataset)
                    #         # lr = nn.functional.mse_loss(curr_logits[:, classes:],
                    #         #                             past_logits[:, classes:])
                    #
                    #         lr = nn.functional.mse_loss(curr_logits,
                    #                                     past_logits)
                    #     elif self.logit_regularization == 'cosine':
                    #         lr = 1 - nn.functional.cosine_similarity(
                    #             curr_logits, past_logits, -1)
                    #         lr = lr.mean()
                    #     else:
                    #         assert False
                    #
                    #     loss += lr * self.gamma * factor

                    # if self.delta > 0:
                    #     classes = len(
                    #         self.experience.classes_in_this_experience)
                    #
                    #     a = curr_features
                    #     b = past_features
                    #
                    #     dist = nn.functional.mse_loss(a, b)
                    #     # dist = 1 - nn.functional.cosine_similarity(a, b, -1)
                    #     dist = dist.mean()
                    #
                    #     loss += dist * self.delta * factor

        # future_reg = 0
        # if self.future_task_reg > 0 and self.future_margin > 0:
        #     future_logits = self.mb_future_logits
        #     # future_mx = future_logits.max(-1).values
        #     # mn = future_logits.min()
        #     # if mn < 0:
        #     #     future_logits = future_logits - mn
        #     future_mx = future_logits.min(-1).values.detach()
        #     current_mx = torch.cat(self.mb_output, -1).max(-1).values
        #
        #     # margins = torch.tensor(self.future_margins, device=self.device)
        #     # reg = current_mx - future_mx - margins[self.mb_task_id]
        #
        #     margin = self.future_margin
        #     # margin = torch.where(current_mx > 0,
        #     #                      - future_mx - margin,
        #     #                      + future_mx - margin)
        #     reg = current_mx - future_mx - margin
        #
        #     # reg = current_mx - future_mx - self.future_margin
        #     reg = torch.relu(reg)
        #     mask = reg > 0
        #     if any(mask):
        #         future_reg = reg[mask].mean()
        #
        #     future_reg = future_reg * self.future_task_reg
        #
        #     loss += future_reg

        return loss
