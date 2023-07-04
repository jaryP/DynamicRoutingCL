import torch
from avalanche.benchmarks import CLExperience
from avalanche.benchmarks.utils import ConstantSequence
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from torch import cosine_similarity, nn


class CumulativeMultiHeadClassifier(MultiTaskModule):

    def __init__(
            self,
            in_features,
            initial_out_features=2,
            masking=True,
            mask_value=-1000,
    ):
        super().__init__()
        self.classes_so_far = 0
        self.in_features = in_features
        self.classifiers = nn.ModuleDict()

    def adaptation(self, experience: CLExperience):

        curr_classes = experience.classes_in_this_experience
        task_labels = experience.task_labels
        # if isinstance(task_labels, ConstantSequence):
        #     # task label is unique. Don't check duplicates.
        #     task_labels = [task_labels[0]]

        tid = str(task_labels[0])

        if tid not in self.classifiers:  # create new head
            new_head = nn.Linear(
                self.in_features, len(curr_classes)
            )
            self.classifiers[tid] = new_head

            # super().adaptation(experience)
            self.classes_so_far += len(curr_classes)

    def forward_single_task(self, x, task_label):
        """compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """

        # if task_label == 0:
        #     task_label = str(task_label)
        #     o = self.classifiers[task_label](x)
        # else:
        out = torch.cat([self.classifiers[str(t)](x)
                       for t in range(task_label + 1)], -1)

        diff = abs(self.classes_so_far - out.shape[-1])

        if diff != 0:
            # out = torch.zeros(len(o), self.classes_so_far, device=o.device)
            out = nn.functional.pad(out, (0, diff))
            # out[:, :o.shape[-1]] = o
            # out[:, o.shape[-1]:] = -torch.inf

            # return out

        return out


def calculate_distance(x, y,
                       distance: str = None,
                       sigma=2,
                       normalize=False):
    if distance is None:
        distance = 'euclidean'

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception(f'{x.shape} {y.shape}')

    if normalize:
        x = torch.nn.functional.normalize(x, 2, -1)
        y = torch.nn.functional.normalize(y, 2, -1)

    a = x.unsqueeze(1).expand(n, m, d)
    b = y.unsqueeze(0).expand(n, m, d)

    if distance == 'euclidean':
        similarity = torch.pow(a - b, 2).sum(2).sqrt()
    elif distance == 'rbf':
        similarity = torch.pow(a - b, 2).sum(2).sqrt()
        similarity = similarity / (2 * sigma ** 2)
        similarity = torch.exp(similarity)
    elif distance == 'cosine':
        similarity = 1 - cosine_similarity(a, b, -1)
    else:
        assert False

    return similarity


def calculate_similarity(x, y,
                         distance: str = None,
                         sigma=2,
                         normalize=False):
    if distance is None:
        distance = 'euclidean'

    di = calculate_distance(x, y, distance, sigma, normalize)

    if distance == 'cosine':
        return 1 - di
    else:
        return - di
