import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as grid_spec
# import pandas as pd
# import seaborn

# plt.rcParams['text.latex.unicode']=True

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

path = '../cifar10/margin_future_reg_debug/debug_margin_logits'

with open(os.path.join(path, 'logits.pkl'), 'rb') as f:
    results = pickle.load(f)

with open(os.path.join(path, 'after_logits.pkl'), 'rb') as f:
    initial_results = pickle.load(f)

# fig = plt.figure(figsize=(8,6))
f, ax = plt.subplots(1, 1, figsize=(8,6))

all_logits = []
for tid, r in results.items():
    logits = [_r[0][1].mean(0) for _r in r]
    logits = np.asarray(logits)

    diff = logits[:, :2].max(1) - logits[:, 2:].max(1)
    all_logits.append(logits)

for i in range(1, 4):
    logits = []

    epochs = []
    ll = results[i][0][0]

    for a, b in results[i].items():
        pass


gs = (grid_spec.GridSpec(len(initial_results),1))
fig = plt.figure(figsize=(8,6))

ax_objs = []

logits = []
for _, task in initial_results.items():
    logits.append(task[0][1].mean(0))

pal = seaborn.cubehelix_palette(10, rot=-.25, light=.7)
g = seaborn.FacetGrid(logits, aspect=15, height=.5, palette=pal)

i = 0
for _, task in initial_results.items():
    task = task[0]

    ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))
    # plot = (task[0].score.plot.kde(ax=ax_objs[-1],color="#f0f0f0", lw=0.5))
    plot = plt.hist(task[0].mean(1), bins=task[0].shape[-1])

    # x, y = plot[-1].get_children()[0].xy
    # x = plot[-1].get_children()[0]._x
    # y = plot.get_children()[0]._y
    # ax_objs[-1].fill_between(x, y)

    # setting uniform x and y lims
    # ax_objs[-1].set_xlim(0, 1)
    # ax_objs[-1].set_ylim(0, 2.2)

    i += 1

plt.tight_layout()
plt.show()

exit()
gt_max = []
gt_max_std = []
past_max = []
past_max_std = []
margins = []
losses = []

for task_results in results.values():
    for l, p, y in task_results:
        a = 0
        correct_max = p[range(len(p)), y]
        pm = p[range(len(p)), :-2].max(-1)
        m = correct_max.mean() - (1 / (l.shape[-1] - 1))

        margins.append(m)
        gt_max.append(correct_max.mean())
        gt_max_std.append(correct_max.std())

        past_max.append(pm.mean())
        past_max_std.append(pm.std())

        l = pm - correct_max + m
        losses.append(l.mean())

gt_max = np.asarray(gt_max)
gt_max_std = np.asarray(gt_max_std)

past_max = np.asarray(past_max)
past_max_std = np.asarray(past_max_std)

x = range(len(gt_max))

f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

a0.plot(x, gt_max, label=r'$p_y(x)$',
        linestyle='dotted',
        markevery=3, marker='s')
a0.fill_between(range(len(gt_max)), gt_max - gt_max_std,
                gt_max + gt_max_std, alpha=0.1)

a0.plot(range(len(gt_max)), past_max, label=r'$\mathrm{M}^{<t}(x)$',
         linestyle='dotted', markevery=3, marker='d')
a0.fill_between(range(len(gt_max)), past_max - past_max_std,
                 past_max + past_max_std, alpha=0.1)

a0.plot(range(len(gt_max)), margins,
         linestyle='dashed', label='Average margin')

a0.vlines([0, 20, 40, 60], -0.1, 1.1, alpha=0.5, colors='k',
           linestyle='dashed')

a0.set_ylabel('Probability')
# plt.xlabel('Epoch')


# plt.savefig("margins.pdf")

# plt.xlabel(r'$\textbf{time (s)}$')

# plt.show()

a1.vlines([0, 20, 40, 60], 0, 0.25, alpha=0.5, colors='k', linestyle='dashed')
a1.set_xlabel('Epoch')
a1.set_xticks(range(0, 80+1, 10), range(20, 100+1, 10))
a0.set_xticks(range(0, 80+1, 10), [])

a1.set_ylabel('Margin loss')

a1.plot(x, np.maximum(losses, 0))

a0.legend()
plt.savefig("margin_loss.pdf")
plt.show()
