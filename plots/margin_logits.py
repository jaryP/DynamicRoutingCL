import os
import pickle
from collections import defaultdict
from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['text.latex.unicode']=True

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

path = '../cifar100/margin_finale_200_mean__nop/debug_margin_logits'

with open(os.path.join(path, 'logits.pkl'), 'rb') as f:
    results = pickle.load(f)

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
