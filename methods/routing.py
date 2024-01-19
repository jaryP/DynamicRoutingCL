from copy import deepcopy

import numpy as np
import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader

from models import RoutingModel
from models.routing.model import CondensedRoutingModel


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


class ContinuosRouting(SupervisedTemplate):
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

        # assert isinstance(model,
        #                   (RoutingModel, CondensedRoutingModel)), (
        #     f'When using '
        #     f'{self.__class__.__name__} the model must be a ContinuosRouting one')

        self.future_margins = []
        self.tasks_nclasses = dict()
        self.current_mb_size = 0

        self.cl_w = 0

        self.layer_wise_regularization = False
        self.double_sampling = -1 if reg_sampling_bs is None else reg_sampling_bs
        self.scheduled_factor = False

        # Past task loss weight
        self.alpha = alpha
        
        self.past_task_reg = past_task_reg
        self.past_margin = past_margin
        self.warm_up_epochs = warm_up_epochs

        self.future_task_reg = future_task_reg
        self.future_margin = future_margin

        # Past task features weight
        self.delta = 0

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

    def _after_forward(self, **kwargs):
        self.mb_output, self.mb_features, self.mb_future_logits, self.mb_future_features = self.mb_output
        self.mb_future_logits = torch.cat(self.mb_future_logits, -1)

        super()._after_forward(**kwargs)

    def _after_eval_forward(self, **kwargs):
        logits, self.mb_features, self.mb_future_logits, self.mb_future_features = self.mb_output
        self.mb_output = torch.cat(logits, -1)
        # self.mb_future_logits = torch.cat(self.mb_future_logits, -1)

        super()._after_forward(**kwargs)

    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        self._after_training_exp_f(**kwargs)

        super()._after_training_exp(**kwargs)

        classes_so_far = len(self.experience.classes_seen_so_far)

        self.model.past_batch = self.sample_past_batch(256)

        if not isinstance(self.model, CondensedRoutingModel):
            return

        if classes_so_far > len(self.experience.classes_in_this_experience):
            dataset = torch.utils.data.ConcatDataset(self.past_dataset.values())

            past_model = deepcopy(self.model).eval()

            new_head = nn.Linear(self.model.backbone_output_dim,
                                 classes_so_far).to(self.device)
            self.model.condensed_model[-1] = new_head

            assigned_paths = self.model.assigned_paths

            optimizer = SGD(self.model.condensed_model.parameters(), lr=0.01)

            datalaoder = DataLoader(dataset, len(dataset) // 20, shuffle=True)
            self.model.train()

            for _ in range(100):
                losses = []
                for x, y, _ in datalaoder:
                    x, y = x.to(self.device), y.to(self.device)

                    with torch.no_grad():
                        past_logits = past_model(x)[0]

                    current_logits = self.model.condensed_model(x)

                    loss = nn.functional.mse_loss(current_logits, past_logits)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

            self.model.reallocate_paths(assigned_paths)
            self.model.use_condensed_model = True

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
        # self.future_margin = self.future_margin + self.margin * 2
        # self.future_margins.append(self.future_margin)

    def criterion(self):
        tid = self.experience.current_experience

        if not self.is_training:
            loss_val = nn.functional.cross_entropy(self.mb_output,
                                                   self.mb_y)

            return loss_val

        past_reg = 0
        future_reg = 0

        # pred = self.mb_output
        #
        # if self.mb_future_logits is not None:
        #     pred = torch.cat((self.mb_output, self.mb_future_logits), 1)

        if len(self.past_dataset) == 0:
            pred = torch.cat(self.mb_output, -1)
            if self.mb_future_logits is not None:
                # pred = torch.cat((pred, self.mb_future_logits), -1)
                pred = torch.cat((pred, self.mb_future_logits.detach()), -1)

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

                    # margin_dist = torch.relu(past_max - mx_current_classes + self.margin)
                    # margin_dist = torch.relu(past_max - mx_current_classes + (1 / (distr.shape[-1] - 1)))
                    margin = mx_current_classes.mean() / 2
                    margin_dist = torch.relu(past_max - mx_current_classes + margin)

                    den_mask = margin_dist > 0
                    margin_den += den_mask.sum().item()

                    past_reg = margin_dist[den_mask].sum()

                    margin_loss += past_reg

                present = torch.cat((co, self.mb_future_logits[mask].detach()), -1)
                # present = co

                loss = nn.functional.cross_entropy(present, y, reduction='sum')

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
                        past_logits = self.past_model(x)[0]
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

        future_reg = 0
        if self.future_task_reg > 0 and self.future_margin > 0:
            future_logits = self.mb_future_logits
            # future_mx = future_logits.max(-1).values
            # mn = future_logits.min()
            # if mn < 0:
            #     future_logits = future_logits - mn
            future_mx = future_logits.min(-1).values.detach()
            current_mx = torch.cat(self.mb_output, -1).max(-1).values

            # margins = torch.tensor(self.future_margins, device=self.device)
            # reg = current_mx - future_mx - margins[self.mb_task_id]

            margin = self.future_margin
            # margin = torch.where(current_mx > 0,
            #                      - future_mx - margin,
            #                      + future_mx - margin)
            reg = current_mx - future_mx - margin

            # reg = current_mx - future_mx - self.future_margin
            reg = torch.relu(reg)
            mask = reg > 0
            if any(mask):
                future_reg = reg[mask].mean()

            future_reg = future_reg * self.future_task_reg

            loss += future_reg

        return loss


class ContinuosRoutingLogits(ContinuosRouting):
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
                 eval_mb_size: int = None,
                 device=None,
                 plugins=None,
                 batch_size_mem=None,
                 criterion=None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 ):

        super().__init__(
            mem_size=mem_size,
            batch_size_mem=batch_size_mem,
            alpha=alpha,
            past_task_reg=past_task_reg,
            past_margin=past_margin,
            warm_up_epochs=warm_up_epochs,
            future_task_reg=future_task_reg,
            future_margin=future_margin,
            gamma=gamma,
            model=model, optimizer=optimizer, criterion=None,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_training_exp(self, **kwargs):
        self._after_training_exp_f(**kwargs)

        super()._after_training_exp(**kwargs)

        classes_so_far = len(self.experience.classes_seen_so_far)

        # self.model.past_batch = self.sample_past_batch(256)
        #
        # if not isinstance(self.model, CondensedRoutingModel):
        #     return

        if classes_so_far > len(self.experience.classes_in_this_experience):
            dataset = torch.utils.data.ConcatDataset(self.past_dataset.values())

            optimizer = SGD(self.model.parameters(), lr=0.01)

            datalaoder = DataLoader(dataset, len(dataset) // 20, shuffle=True)
            self.model.train()

            for _ in range(100):
                losses = []
                for x, y, _ in datalaoder:
                    x, y = x.to(self.device), y.to(self.device)

                    current_logits = self.model(x)[0]

                    loss = nn.functional.cross_entropy(current_logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

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

        x, y, t, logits = [], [], [], []

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            ot_x, ot_y, ot_tid, ot_l = next(iter(DataLoader(d,
                                                            batch_size=bs,
                                                            shuffle=True)))

            x.append(ot_x)
            y.append(ot_y)
            t.append(ot_tid)
            logits.append(ot_l)

        return (torch.cat(x, 0), torch.cat(y, 0),
                torch.cat(t, 0), torch.cat(logits, 0))

    def _before_forward_f(self, **kwargs):

        super()._before_forward(**kwargs)

        if len(self.past_dataset) == 0:
            return None

        x, y, t, l = self.sample_past_batch(len(self.mb_x))
        self.current_mb_size = len(self.mb_x)

        self.mbatch[0] = torch.cat((self.mbatch[0], x.to(self.device)))
        self.mbatch[1] = torch.cat((self.mbatch[1], y.to(self.device)))
        self.mbatch[2] = torch.cat((self.mbatch[2], t.to(self.device)))
        self.past_logits = l

    @torch.no_grad()
    def _after_training_exp_f(self, **kwargs):

        samples_to_save = self.memory_size // len(
            self.experience.classes_seen_so_far)

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)

                all_logits = []
                for x, y, _, l in DataLoader(d, batch_size=self.train_mb_size):
                    _l = self.model(x.to(self.device))[0]
                    _l = _l.cpu()
                    l = torch.cat((l, _l[:, l.shape[-1]:]), -1)
                    all_logits.append(l)

                all_logits = torch.cat(all_logits, 0)
                d.logits = all_logits

                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        self.model.eval()
        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)
            all_logits = []
            for x, _, _ in DataLoader(dataset.train().subset(indexes),
                                      batch_size=self.eval_mb_size):
                x = x.to(self.device)

                logits = self.model(x)[0]
                all_logits.append(logits.cpu())

            all_logits = torch.cat(all_logits, 0)

            d = LogitsDataset(dataset.train().subset(indexes), all_logits)

            self.past_dataset[y] = d

        # if self.experience.current_experience > 0:
        self.past_model = deepcopy(self.model)
        self.past_model.eval()

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        self._before_forward_f(**kwargs)

    def criterion(self):
        if not self.is_training:
            if isinstance(self.model, RoutingModel):
                loss_val = nn.functional.cross_entropy(self.mb_output,
                                                       self.mb_y)
            else:
                log_p_y = torch.log_softmax(self.mb_output, dim=1)
                loss_val = -log_p_y.gather(1, self.mb_y.unsqueeze(-1)).squeeze(
                    -1).mean()

            return loss_val

        pred = self.mb_output

        if self.mb_future_logits is not None:
            pred = torch.cat((self.mb_output, self.mb_future_logits), 1)

        if len(self.past_dataset) == 0:
            if isinstance(self.model, RoutingModel):
                loss = nn.functional.cross_entropy(pred, self.mb_y,
                                                   label_smoothing=0)
            else:
                log_p_y = torch.log_softmax(pred, dim=1)
                loss = -log_p_y.gather(1, self.mb_y.unsqueeze(-1)).squeeze(
                    -1).mean()
        else:
            s = self.current_mb_size
            s1 = len(self.mb_x) - s

            pred1, pred2 = torch.split(pred, [s, s1], 0)
            y1, y2 = torch.split(self.mb_y, [s, s1], 0)

            y1 = y1 - min(
                self.experience.classes_in_this_experience)  # min(classes_in_this_experience)

            neg_pred1 = pred1[:, :len(self.experience.previous_classes)]
            pred1 = pred1[:, len(self.experience.previous_classes):]

            if isinstance(self.model, RoutingModel):
                loss1 = nn.functional.cross_entropy(pred1, y1)
                loss2 = nn.functional.cross_entropy(pred2, y2)
            else:
                pred1 = torch.log_softmax(pred1, 1)
                pred2 = torch.log_softmax(pred2, 1)
                loss1 = -pred1.gather(1, y1.unsqueeze(-1)).squeeze(-1).mean()
                loss2 = -pred2.gather(1, y2.unsqueeze(-1)).squeeze(-1).mean()

            loss = loss1 + self.alpha * loss2

            w = 1
            if self.warm_up_epochs > 0:
                w = float(
                    self.clock.train_exp_epochs / self.warm_up_epochs >= 1)
                # w = self.clock.train_exp_epochs / self.warm_up_epochs

            if w >= 1 and self.past_margin > 0 and self.past_task_reg > 0:
                mx = neg_pred1.max(-1).values.detach()
                # mx = neg_pred1.max(-1).values
                mx_current_classes = pred1[range(len(pred1)), y1]

                margin_dist = torch.maximum(torch.zeros_like(mx),
                                            mx - mx_current_classes + self.past_margin)

                past_reg = margin_dist.mean()

                past_reg = past_reg * self.past_task_reg
                loss += past_reg

            if self.is_training and any([self.gamma > 0, self.delta > 0]):
                if self.scheduled_factor:
                    current_classes = len(
                        self.experience.classes_in_this_experience)
                    seen_classes = len(self.experience.classes_seen_so_far)

                    factor = (seen_classes / current_classes) ** 0.5
                else:
                    factor = 1

                if self.past_model is not None:
                    bs = len(self.mb_output) // 2

                    if self.double_sampling:
                        x, y, _, logits = self.sample_past_batch(bs)
                        x, y = x.to(self.device), y.to(self.device)
                        past_logits = logits.to(self.device)

                        curr_logits, curr_features, random_logits, _ = self.model(
                            x)
                        curr_logits = torch.cat((curr_logits, random_logits),
                                                -1)
                    else:
                        # x, y = (self.mb_x[self.current_mb_size:],
                        #         self.mb_y[self.current_mb_size:])
                        # curr_logits = self.mb_output[ self.current_mb_size:]
                        # curr_features = self.mb_features[self.current_mb_size:]

                        curr_logits = pred[self.current_mb_size:]
                        past_logits = self.past_logits
                        # curr_features = self.mb_features[ self.current_mb_size:]

                    # with torch.no_grad():
                    #     # past_logits, past_features, _, _ = self.past_model(x, other_paths=self.model.current_random_paths)
                    #     past_logits, past_features, _, _ = self.past_model(x,
                    #                                                        other_paths=None)
                    #     curr_logits = curr_logits[:, :past_logits.shape[1]]

                    if self.gamma > 0:
                        if self.logit_regularization == 'kl':
                            curr_logits = torch.log_softmax(
                                curr_logits / self.tau, -1)
                            past_logits = torch.log_softmax(
                                past_logits / self.tau, -1)

                            lr = nn.functional.kl_div(curr_logits,
                                                      past_logits,
                                                      log_target=True,
                                                      reduction='batchmean')
                        elif self.logit_regularization == 'mse':
                            classes = len(self.past_dataset)
                            # lr = nn.functional.mse_loss(curr_logits[:, classes:],
                            #                             past_logits[:, classes:])
                            lr = nn.functional.mse_loss(
                                curr_logits[:, :past_logits.shape[1]],
                                past_logits)
                        elif self.logit_regularization == 'cosine':
                            lr = 1 - nn.functional.cosine_similarity(
                                curr_logits, past_logits, -1)
                            lr = lr.mean()
                        else:
                            assert False

                        loss += lr * self.gamma * factor

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

                # elif self.ot_logits is not None:
                #     bs = len(self.mb_output) // 2
                #
                #     past_logits = self.ot_logits
                #
                #     if self.gamma > 0:
                #         curr_logits = self.mb_output[bs:]
                #
                #         lr = nn.functional.mse_loss(curr_logits, past_logits,
                #                                     reduction='mean')
                #
                #         loss += lr * self.gamma

        if self.future_task_reg > 0 and self.future_margin > 0:
            future_logits = self.mb_future_logits
            future_mx = future_logits.max(-1).values
            current_mx = self.mb_output.max(-1).values

            reg = current_mx - future_mx - self.future_margin
            reg = torch.maximum(torch.zeros_like(reg), reg)
            future_reg = reg.mean() * self.future_task_reg

            loss += future_reg

        return loss
