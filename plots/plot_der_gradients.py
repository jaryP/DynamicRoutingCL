import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from matplotlib.patches import Patch


def draw_plot(data, ax, offset, edge_color, fill_color):
    pos = np.arange(data.shape[1])+offset
    # violin = ax.boxplot(data, positions= pos, widths=0.05, patch_artist=False)
    violin = ax.violinplot(data, positions= pos, widths=0.15, showmedians=True, showextrema= False)
    return violin

    # for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    #     plt.setp(bp[element], color=edge_color)
    # for patch in bp['boxes']:
    #     patch.set(facecolor=fill_color)

path = '../der/debug_gradients/debug/'

with open(os.path.join(path, 'gradients.pkl'), 'rb') as f:
    results = pickle.load(f)

print(results[0][0].keys(), results[0][1].keys())

all_statistics = []
par_distances = []
model_distances = []

names = ['classifier.weight',
         'feature_extractor.conv1.weight',
         'feature_extractor.layer1.2.conv2.weight',
         'feature_extractor.layer2.2.conv2.weight',
         'feature_extractor.layer3.2.conv2.weight']

for gm, gmse in results:
    statistics = []
    distance = []

    for n, v in gm.items():
        if len(v.shape) > 1 and n in names:
            d = torch.abs(v.flatten() - gmse[n].flatten())
            distance.append(( torch.nn.functional.cosine_similarity(v.flatten(1), gmse[n].flatten(1), -1)).cpu().numpy())
            statistics.append((d.min().item(), d.max().item(), d.mean().item(), d.std().item()))
    # distances = [torch.max(v.flatten() - gmse[n].flatten()).item() for n, v in gm.items() if len(v.shape) > 1]
    all_statistics.append(statistics)
    par_distances.append(distance)

all_statistics = np.asarray(all_statistics)
par_distances = list(zip(*par_distances))

for gm, gmse in results:
    a = torch.cat([p.flatten() for k, p in gm.items()])
    b = torch.cat([p.flatten() for k, p in gmse.items()])

    dist = torch.nn.functional.cosine_similarity(a, b, 0)
    model_distances.append(dist)

epochs = np.asarray([4, 9, 14, 19])
offsets = [-0.35, -0.2, 0, 0.2, 0.35]

labels = []
fig, ax = plt.subplots()

legens_labels = ['Input Conv. Layer',
                 'Conv. Layer block 1',
                 'Conv. Layer block 2',
                 'Conv. Layer block 3',
                 'Classification Layer']

for i in range(len(offsets)):
    violin = draw_plot(np.asarray(par_distances[i])[epochs].T, ax, offsets[i], "tomato", "white")
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((Patch(color=color), legens_labels[i]))

    # xy = [[l.vertices[:,0].mean(),l.vertices[0,1]] for l in violin['cmedians'].get_paths()]
    # xy = np.array(xy)
    # ax.scatter(xy[:, 0], xy[:, 1], s=100, c="red", marker="o", zorder=3)


# draw_plot(np.asarray(par_distances[1])[epochs].T, ax, +0.2,"skyblue", "white")
ax.set_xticks(np.arange(len(epochs)), epochs+1)
plt.xlabel('Epoch')
plt.ylabel('Gradients Cosine Similarity')
plt.legend(*zip(*labels), loc='lower left', ncols=2)
plt.show()

fig.savefig('violins.pdf',  bbox_inches='tight')

exit()

a = 0

names = [n for n, v in results[0][0].items() if len(v.shape) > 1]
x = np.arange(all_statistics.shape[0])

for i in range(0, len(all_statistics), 5):
    y = all_statistics[:, i, -2]
    plt.plot(x, y)
    plt.errorbar(x, y - all_statistics[:, i, -1], y + all_statistics[:, i, -1])
# plt.legend()
plt.show()

# p_margin_mean, p_margin_std = np.asarray([0.1887, 0.0394]), [0.1425, 0.1346]
# l_margin_mean, l_margin_std = np.asarray([0.8125, 0.1875]), [0.3966, 0.3966]
# l_margin_2 = [], []
#
# logits = np.asarray([3.6362, -3.3722,  0.2788, -0.5738])
# probs = np.asarray([0.8665, 0.0451, 0.0569, 0.0315])
#
#
# fig, ax = plt.subplots(1)
#
# x = np.asarray([0, 1])
# width = 0.25  # the width of the bars
#
# results = np.stack((p_margin_mean, l_margin_mean), 0)
#
# print(results)
# offset = width * 0
# rects = ax.bar(x, results[0, :], width, label='Margin on probabilities (m=0.3)')
# # ax.bar_label(rects, padding=3)
#
# rects = ax.bar(x + width, results[1, :], width,  label='Margin on logits (m=1)')
# # ax.bar_label(rects, padding=3)
#
# ax.set_xticks(x + (width / 2), ['Class 0', 'Class 1'])
# ax.legend(loc='upper right', ncols=1)
# ax.set_ylabel('Gradients magnitude')
#
# fig.savefig('gradients.pdf',  bbox_inches='tight')
# plt.show()
#
#
# fig, (ax1, ax2) = plt.subplots(2, 1)
#
# x = np.asarray([0, 1, 3, 4], dtype='float')
# width = 0.1  # the width of the bars
#
# results = np.stack((p_margin_mean, l_margin_mean), 0)
# color = ['red', 'blue', 'purple', 'red']
# legend = ['Task 0', 'Task 1']
#
# print(results)
# offset = width * 0
# rects = ax1.bar(x[:2], logits[:2], width=1.0, alpha=0.8, color='red')
# rects = ax1.bar(x[2:], logits[2:], width=1.0, alpha=0.8, color='blue')
# ax1.hlines(0, -0.48, 4.48, colors='k', linewidths=0.8)
# # ax.bar_label(rects, padding=3)
# ax1.set_xticks([], [])
#
# # rects = ax2.bar(x, probs, width=1.0, color=color)
# rects = ax2.bar(x[:2], probs[:2], width=1.0, alpha=0.8, color='red')
# rects = ax2.bar(x[2:], probs[2:], width=1.0, alpha=0.8,  color='blue')
# # ax.bar_label(rects, padding=3)
#
# ax2.set_xticks(x + (width / 2), ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
# # plt.legend(loc='upper right', ncols=1)
# # ax.set_ylabel('Gradients magnitude')
#
# handles = [plt.Rectangle((0,0),1,1, color=c) for c in ['red', 'blue']]
# plt.legend(handles, ['Classes from task 0', 'Classes from task 1'])
# fig.savefig('logits.pdf',  bbox_inches='tight')
#
# plt.show()