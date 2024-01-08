import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as grid_spec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

path = '/media/jary/Data/progetti/DynamicModelCL/cifar10/no_test_bias/0/debug_logits'
paths = ['/media/jary/Data/progetti/DynamicModelCL/cifar10/no_test_bias/0/debug_logits',
         '/media/jary/Data/progetti/DynamicModelCL/cifar10/plot_probs/0/logits/',
         '/media/jary/Data/progetti/DynamicModelCL/cifar10/_provadist_logits/0/debug_logits']

markers = ['x', 's', '^']
legends = ['Our proposal', 'DER', 'LD']

f, axs = plt.subplots(4, 1, figsize=(8, 6))

for pi, p in enumerate(paths):
    with open(os.path.join(p, 'logits.pkl'), 'rb') as f:
        results = pickle.load(f)

    max_len = (len(results[0]) - 1) * (len(results) - 1)

    for t in results:
        diffs = []
        for task, epochs in results.items():
            if task <= t:
                continue
            labels = epochs[0][t][2]
            mx = max(labels)

            probs = np.asarray([e[t][1] for e in epochs])
            initial_prob, probs = probs[0], probs[1:]
            correct_prob = np.take_along_axis(probs, labels[None, :, None], 2).squeeze(-1)
            future_max_prob = probs[:, :, mx+1:].max(-1)

            distance = correct_prob - future_max_prob
            # d = initial_prob * np.log(initial_prob / probs)
            # d[np.logical_or(np.isinf(d), np.isnan(d))] = 0.0
            # distance = np.sum(d, -1)

            # initial_distance, distance = distance[0], distance[1:]
            # distance = initial_distance - distance

            diffs.extend(distance)
            # s, probs = probs[0], probs[1:]
            # diff = s - probs

        if len(diffs) == 0:
            continue
        diffs = np.asarray(diffs)
        mn = diffs.mean(1)
        std = diffs.std(1)

        x = range(max_len - len(diffs), max_len)
        axs[t].plot(x, mn, marker=markers[pi], markevery=3, label=legends[pi])
        axs[t].fill_between(x, mn - std, mn + std, alpha=0.1)


for axi, ax in enumerate(axs):
    ax.set_xlim(-1, 80)
    ax.set_ylim(0, 5)
    ax.set_xticks([])
    ax.set_ylabel(f'Task {axi + 1}')
    ax.vlines([0, 20, 40, 60, 79], 0, 10, alpha=0.5, colors='k',
               linestyle='dashed')

    # axins = zoomed_inset_axes(ax, zoom=0.5, loc='upper center')
    #
    # asb = AnchoredSizeBar(ax.transData,
    #                       size,
    #                       str(size),
    #                       loc=8,
    #                       pad=0.1, borderpad=0.5, sep=5,
    #                       frameon=False)
    # ax.add_artist(asb)


axs[-1].set_xticks(range(0, 80+1, 5), range(20, 100+1, 5))
axs[-1].set_xlabel(f'Epoch')
axs[-1].legend()

plt.savefig('future.pdf',  bbox_inches='tight')
plt.show()
