import os
import pickle
from collections import defaultdict
from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph
import networkx as nx

path = '/media/jary/Data/progetti/DynamicModelCL/results/splitcifar10/Replay/replay_200__cil_cifar10_5__False__1__False__incremental__resnet20__sgd__20/debug/'

with open(os.path.join(path, 'logits.pkl'), 'rb') as f:
    results = pickle.load(f)

with open(os.path.join(path, 'eval_scores.pkl'), 'rb') as f:
    eval_results = pickle.load(f)

all_differences = []
replay_scores = []
average_logits = []
tid = []

all_task_results = defaultdict(list)
all_task_gt = defaultdict(list)

classes_tasks = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

# TODO: usare le label dei task per accorpore i punteggi
for task_results in results.values():
    initial_logits = task_results[0][0]

    A = kneighbors_graph(initial_logits, 2).toarray()
    G = nx.from_numpy_array(A)
    L = nx.normalized_laplacian_matrix(G)
    ie = np.linalg.eigvals(L.toarray())
    ie = sorted(ie)

    ieigenvalues, ieigenvectors = np.linalg.eig(L.toarray())

    # e = np.asarray(e)

    # plt.plot(range(len(e)), e)
    # plt.show()

    # plt.hist(np.asarray(e), bins=100)  # histogram with 100 bins
    # plt.xlim(0, 2)  # eigenvalues between 0 and 2
    # plt.show()

    distances = []

    for i in range(1, len(task_results)):
        l = task_results[i][0]
        A = kneighbors_graph(l, 2).toarray()
        G = nx.from_numpy_array(A)
        L = nx.normalized_laplacian_matrix(G)
        e = np.linalg.eigvals(L.toarray())
        e = sorted(e)
        eigenvalues, eigenvectors = np.linalg.eig(L.toarray())
        # e = np.asarray(e)

        distances.append(np.linalg.norm(ieigenvectors - eigenvectors, 2, -1).mean())

    plt.plot(range(len(distances)), distances)
    plt.show()

    # D = np.diag(A.sum(1))
    # L = D - A
    #
    # N = np.power(D, -0.5) * L * np.power(D, -0.5)
    # np.linalg.eigvals(L.toarray())

    initial_score = (initial_logits.argmax(-1) == task_results[0][-1]).mean()

    training_results = [np.linalg.norm(task_results[i][0] - initial_logits, 2, -1) for i in range(1, len(task_results))]
    # replay_scores.extend([(task_results[i][0].argmax(-1) == task_results[i][-1]).mean() - initial_score for i in range(1, len(task_results))])
    replay_scores.extend([(task_results[i][0].argmax(-1) == task_results[i][-1]).mean() for i in range(1, len(task_results))])

    average_logits.extend([task_results[i][0][range(200), task_results[i][-1]].mean() for i in range(1, len(task_results))])

    all_differences.extend(np.mean(training_results, 1))
    tid.append(classes_tasks[task_results[0][-1]])

    logits, probs, gt = list(zip(*task_results[1:]))

    tids = classes_tasks[task_results[0][-1]]

    for t in np.unique(tids):
        mask = tids == t
        for l, g in zip(logits, gt):
            all_task_results[t].append(l[mask])
            all_task_gt[t].append(g[mask])

    logits = np.stack(logits)
    # for i, t in enumerate(classes_tasks[task_results[0][-1]]):
    #     all_task_results[t].append(task_results[i])

exit()
max_epochs = max(map(len, all_task_results.values()))
fig, (ax1, ax2, ax3) = plt.subplots(3)

markers = ['.', 's', 'x', 'd', 'v']

for t in all_task_results.keys():
    l = all_task_results[t]
    gt = all_task_gt[t]

    correct_logits = []
    correct_errors = []

    memory_scores = []

    for g, l in zip(gt, l):
        correct_logits.append(l[range(len(l)), g].mean())
        correct_errors.append(l[range(len(l)), g].std())

        score = (l.argmax(-1) == g).mean()
        memory_scores.append(score)

    correct_logits = np.asarray(correct_logits)
    correct_errors = np.asarray(correct_errors)

    s = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{t:03}'
    eval_scores = eval_results[s][1]
    _eval_scores = []

    for i in range(len(eval_scores) % 20):
        _eval_scores.extend(eval_scores[(i * 20): (i * 20) + 20])
        # eval_scores.pop(i * 20)

    if len(_eval_scores) > 0:
        eval_scores = _eval_scores
        if t != 4:
            eval_scores = eval_scores[20:]

    x = range(max_epochs - len(correct_logits), max_epochs)
    ax1.plot(x, correct_logits, alpha=0.5, marker=markers[t],
             linestyle='dashed', markevery=3, label=f'Task {t + 1}')
    # ax1.errorbar(x, correct_logits, yerr=correct_errors)
    ax1.fill_between(x, correct_logits - correct_errors, correct_logits + correct_errors, alpha=0.1)

    ax2.plot(x, memory_scores, marker=markers[t], linestyle='dashed', markevery=3)

    ax3.plot(x, eval_scores, marker=markers[t], linestyle='dashed', markevery=3)

# ax1.set_xticks([])
ax1.legend(ncols=2)
ax1.set_xticks(np.arange(0, 80+1, 5), [])
ax1.vlines([0, 20, 40, 60], 0, 20, alpha=0.5, colors='k', linestyle='dashed')
# ax1.set_ylabel('Logits magnitude')

# ax2.set_xticks([])
# ax2.set_ylabel('Memory accuracy')
ax2.set_xticks(np.arange(0, 80+1, 5), [])
ax2.vlines([0, 20, 40, 60], 0, 1, alpha=0.5, colors='k', linestyle='dashed')

ax3.set_xticks(np.arange(0, 80+1, 5), np.arange(20, 100+1, 5))
# ax3.set_ylabel('Test accuracy')
ax3.set_xlabel('Epoch')
ax3.vlines([0, 20, 40, 60], 0, 1, alpha=0.5, colors='k', linestyle='dashed')

plt.show()

fig.savefig(
    "eval_score.pdf",
    bbox_inches=mtransforms.Bbox([[0, 0], [1, 0.35]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)

fig.savefig(
    "memory_score.pdf",
    bbox_inches=mtransforms.Bbox([[0, 0.35], [1, 0.63]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)

fig.savefig(
    "logits_magnitude.pdf",
    bbox_inches=mtransforms.Bbox([[0, 0.62], [1, 0.9]]).transformed(
        fig.transFigure - fig.dpi_scale_trans
    ),
)
