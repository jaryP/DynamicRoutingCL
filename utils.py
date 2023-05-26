import torch
from torch import cosine_similarity


def calculate_distance(x, y,
                       distance: str = None,
                       sigma=2):

    if distance is None:
        distance = 'euclidean'

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception(f'{x.shape} {y.shape}')

    a = x.unsqueeze(1).expand(n, m, d)
    b = y.unsqueeze(0).expand(n, m, d)

    if distance == 'euclidean':
        a = torch.nn.functional.normalize(a, 2, -1)
        b = torch.nn.functional.normalize(b, 2, -1)
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
                         sigma=2):
    if distance is None:
        distance = 'cosine'

    di = calculate_distance(x, y, distance, sigma)

    if distance == 'cosine':
        return 1 - di
    else:
        return - di
