import os
import pickle
from collections import defaultdict
from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np

path = '/media/jary/Data/progetti/DynamicModelCL/results/splitcifar10/Replay/replay_200__cil_cifar10_5__False__1__False__incremental__resnet20__sgd__20/debug/'

with open(os.path.join(path, 'logits.pkl'), 'rb') as f:
    results = pickle.load(f)

with open(os.path.join(path, 'eval_scores.pkl'), 'rb') as f:
    eval_results = pickle.load(f)

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

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

# extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ax1_figure.png', bbox_inches=extent.expanded(1.2, 1.1))
#
# extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ax2_figure.png', bbox_inches=extent.expanded(1.2, 1.1))
#
# extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ax3_figure.png', bbox_inches=extent.expanded(1.2, 1.5))

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

exit()
fig, (ax1, ax2, ax3) = plt.subplots(3)

epochs = np.arange(5, len(all_differences)+5)
ax1.plot(epochs, all_differences)
ax2.plot(epochs, replay_scores)
ax3.plot(epochs, average_logits)
# plt.vlines([20, 40, 60, 80], 0, 5)
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks(epochs, epochs)

plt.show()

exit()

mean_logits = []
std_logits = []
past_mean_logits = []
past_std_logits = []

last_logits = {}

mean_probs = []

epochs = np.arange(len(results))
tasks = epochs // 20
classes = [c for c in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)] for _ in range(20)]

for i in [20, 40, 60, 80]:
    i = i - 1
    l, _, gt = results[i]
    mask = np.asarray(list(map(lambda x: np.any(x in classes[i]), gt)))
    l = l[mask]
    last_logits[classes[i]] = l

# final_values = [results[i][0][np.asarray([np.any(results[i][-1] in classes[i]) for g in gt])] for i in [20, 40, 60, 80]]

for i, (l, p, gt) in enumerate(results):
    mask = np.asarray([np.any(g in classes[i]) for g in gt])
    past_mask = np.asarray([np.any(g not in classes[i]) for g in gt])

    l1 = l[range(len(l)), gt][mask]
    mean_logits.append(l1.mean(0))
    std_logits.append(l1.std(0))

    l = l[range(len(l)), gt][past_mask]
    past_mean_logits.append(l.mean(0))
    past_std_logits.append(l.std(0))

    p = p[range(len(p)), gt]
    mean_probs.append(p.mean(0))

print(mean_logits)
plt.plot(epochs, mean_logits)
plt.plot(epochs, past_mean_logits)
# plt.vlines([20, 40, 60, 80], 0, 5)
plt.xticks(epochs)
plt.show()
