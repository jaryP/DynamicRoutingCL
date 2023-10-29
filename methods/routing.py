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
                 eval_mb_size: int = None,
                 device=None,
                 plugins=None,
                 batch_size_mem=None,
                 criterion=None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 ):

        assert isinstance(model,
                          (RoutingModel, CondensedRoutingModel)), f'When using {self.__class__.__name__} the model must be a ContinuosRouting one'
        
        self.tasks_nclasses = dict()
        self.current_mb_size = 0

        self.cl_w = 0

        self.layer_wise_regularization = False
        self.double_sampling = False

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
        self.logit_regularization = 'mse'
        self.tau = 1

        self.memory_size = mem_size

        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem

        self.is_finetuning = False
        self.mb_future_features = None
        self.ot_logits = None

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
        super()._after_forward(**kwargs)

    def _after_eval_forward(self, **kwargs):
        self.mb_output, self.mb_features, self.mb_future_logits, self.mb_future_features = self.mb_output
        super()._after_forward(**kwargs)

    def _after_training_iteration(self, **kwargs):
        self.ot_logits = None
        super()._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        self._after_training_exp_f(**kwargs)

        super()._after_training_exp(**kwargs)

        classes_so_far = len(self.experience.classes_seen_so_far)

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

    def sample_past_batch(self, batch_size):
        if len(self.past_dataset) == 0:
            return None

        classes = (set(self.experience.classes_seen_so_far) -
                   set(self.experience.classes_in_this_experience))

        if len(classes) == 0:
            return None

        samples_per_task = batch_size // len(classes)
        rest = batch_size % len(classes)

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

        bs = len(self.mb_x)
        self.current_mb_size = bs

        classes = (set(self.experience.classes_seen_so_far) -
                   set(self.experience.classes_in_this_experience))

        if len(classes) == 0:
            return

        samples_per_task = max(bs // len(classes), 1)
        rest = bs % len(classes)

        if rest > 0 and samples_per_task > 1:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            ot_x, ot_y, ot_tid = next(iter(DataLoader(d,
                                                      batch_size=bs,
                                                      shuffle=True)))

            self.mbatch[0] = torch.cat((self.mbatch[0], ot_x.to(self.device)))
            self.mbatch[1] = torch.cat((self.mbatch[1], ot_y.to(self.device)))
            self.mbatch[2] = torch.cat((self.mbatch[2], ot_tid.to(self.device)))

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

        return

        all_logits = []
        features = []
        ys = []

        indexes = defaultdict(list)
        classes_count = defaultdict(int)
        features = []

        with torch.no_grad():
            for i, (x, y, t) in enumerate(DataLoader(dataset, batch_size=32)):
                x = x.to(device)

                ys.append(y)
                y = y.to(device)

                f = self.model(x)[1]

                features.append(f[range(len(f)), y].cpu())

            ys = torch.cat(ys, 0)
            features = torch.cat(features, 0)

            ys = np.asarray(ys.numpy())
            features = np.asarray(features.numpy())

            classes_count = collections.Counter(ys)
            tot = sum(classes_count.values())
            classes_count = {k: v / tot for k, v in classes_count.most_common()}

            for y in np.unique(ys):
                mask = ys == y
                f = features[mask]
                indexes = np.argwhere(mask).reshape(-1)

                km = KMeans(n_clusters=4).fit(f)
                # km = AffinityPropagation().fit(f)
                centroids = km.cluster_centers_
                centroids = torch.tensor(centroids)

                f = torch.tensor(f)
                distances = calculate_distance(f, centroids)
                closest_one = distances.argmin(-1).numpy()

                unique_centroids = np.unique(closest_one)
                sample_per_centroid = int(
                    samples_to_save * classes_count[y]) // len(unique_centroids)
                sample_per_centroid = samples_to_save // len(unique_centroids)

                selected_indexes = []

                for i in unique_centroids:
                    mask = closest_one == i
                    _indexes = indexes[mask]

                    selected = np.random.choice(_indexes,
                                                min(sample_per_centroid,
                                                    len(_indexes)),
                                                False)

                    selected_indexes.extend(selected.tolist())

                self.past_dataset[y] = dataset.train().subset(selected_indexes)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        self._before_forward_f(**kwargs)

    @torch.no_grad()
    def _before_training_exp(self, **kwargs):

        super()._before_training_exp(**kwargs)

        if self.model is not None:
            # if self.experience.current_experience > 0:
            #     self.model.internal_features = None
            self.past_model = deepcopy(self.model)
            self.past_model.eval()

    def criterion(self):
        # tid = self.experience.current_experience

        if not self.is_training:
            if isinstance(self.model, RoutingModel):
                loss_val = nn.functional.cross_entropy(self.mb_output,
                                                       self.mb_y)
            else:
                log_p_y = torch.log_softmax(self.mb_output, dim=1)
                loss_val = -log_p_y.gather(1, self.mb_y.unsqueeze(-1)).squeeze(
                    -1).mean()

            return loss_val

        past_reg = 0
        future_reg = 0

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

            w = 1
            if self.warm_up_epochs > 0:
                w = float(
                    self.clock.train_exp_epochs / self.warm_up_epochs >= 1)
                # w = self.clock.train_exp_epochs / self.warm_up_epochs

            if w >= 1 and self.past_margin > 0 and self.past_task_reg > 0:
                # mx = neg_pred1.max(-1).values.detach()
                mx = neg_pred1.max(-1).values
                mx_current_classes = pred1[range(len(pred1)), y1]

                margin_dist = torch.maximum(torch.zeros_like(mx),
                                            mx - mx_current_classes + self.past_margin)

                past_reg = margin_dist.mean()

                past_reg = past_reg * self.past_task_reg

            if isinstance(self.model, RoutingModel):
                loss1 = nn.functional.cross_entropy(pred1, y1)
                loss2 = nn.functional.cross_entropy(pred2, y2)
            else:
                pred1 = torch.log_softmax(pred1, 1)
                pred2 = torch.log_softmax(pred2, 1)
                loss1 = -pred1.gather(1, y1.unsqueeze(-1)).squeeze(-1).mean()
                loss2 = -pred2.gather(1, y2.unsqueeze(-1)).squeeze(-1).mean()

            loss_val = loss1 + self.alpha * loss2

            if self.future_task_reg > 0 and self.future_margin > 0:
                future_logits = self.mb_future_logits
                future_mx = future_logits.max(-1).values
                current_mx = self.mb_output.max(-1).values

                reg = current_mx - future_mx - self.future_margin
                reg = torch.maximum(torch.zeros_like(reg), reg)
                future_reg = reg.mean() * self.future_task_reg

            loss = loss_val + past_reg + future_reg

            if self.is_training and any([self.gamma > 0, self.delta > 0]):
                if self.past_model is not None:
                    bs = len(self.mb_output) // 2

                    if self.double_sampling:
                        x, y, _ = self.sample_past_batch(bs)
                        x, y = x.to(self.device), y.to(self.device)
                        curr_logits, curr_features = self.model(x)[:2]
                    else:
                        x, y = (self.mb_x[self.current_mb_size:],
                                self.mb_y[self.current_mb_size:])
                        # curr_logits = self.mb_output[ self.current_mb_size:]
                        curr_features = self.mb_features[self.current_mb_size:]

                        curr_logits = pred[self.current_mb_size:]
                        # curr_features = self.mb_features[ self.current_mb_size:]

                    with torch.no_grad():
                        past_logits, past_features, _, _ = self.past_model(x,
                                                                           other_paths=self.model.current_random_paths)
                        # past_logits, past_features, _, _ = self.past_model(x, other_paths=None)

                    if self.gamma > 0:
                        if self.layer_wise_regularization:
                            all_lr = 0

                            csf = len(self.experience.previous_classes)
                            for cc, pp in zip(self.model.internal_features[:csf],
                                              self.past_model.internal_features[
                                              :csf]):

                                for c, p in zip(cc, pp):
                                    c = torch.flatten(c, 1)[bs:]
                                    p = torch.flatten(p, 1)

                                    if self.logit_regularization == 'mse':
                                        lr = nn.functional.mse_loss(c, p)
                                    elif self.logit_regularization == 'cosine':
                                        lr = 1 - nn.functional.cosine_similarity(c,
                                                                                 p,
                                                                                 -1)
                                        lr = lr.mean()
                                    else:
                                        assert False

                                    all_lr += lr

                            lr = (all_lr / csf).mean()

                        else:
                            if self.logit_regularization == 'kl':
                                curr_logits = torch.log_softmax(
                                    curr_logits / self.tau, -1)
                                past_logits = torch.log_softmax(
                                    past_logits / self.tau, -1)

                                lr = nn.functional.kl_div(curr_logits, past_logits,
                                                          log_target=True,
                                                          reduction='batchmean')
                            elif self.logit_regularization == 'mse':
                                # lr = nn.functional.mse_loss(curr_logits[:, :-classes],
                                #                             past_logits[:, :-classes])
                                lr = nn.functional.mse_loss(curr_logits,
                                                            past_logits)
                            elif self.logit_regularization == 'cosine':
                                lr = 1 - nn.functional.cosine_similarity(
                                    curr_logits, past_logits, -1)
                                lr = lr.mean()
                            else:
                                assert False

                        loss += lr * self.gamma

                    if self.delta > 0:
                        classes = len(self.experience.classes_in_this_experience)

                        curr_features = curr_features
                        past_features = past_features

                        dist = nn.functional.mse_loss(curr_features, past_features)

                        loss += dist * self.delta

                elif self.ot_logits is not None:
                    bs = len(self.mb_output) // 2

                    past_logits = self.ot_logits

                    if self.gamma > 0:
                        curr_logits = self.mb_output[bs:]

                        lr = nn.functional.mse_loss(curr_logits, past_logits,
                                                    reduction='mean')

                        loss += lr * self.gamma

        return loss
