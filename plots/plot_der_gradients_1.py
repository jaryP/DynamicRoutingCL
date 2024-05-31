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

path = '/media/jary/Data/progetti/DynamicModelCL/cifar10/der/0/debug'

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
        # if len(v.shape) > 1 and n in names:
        if len(v.shape) > 1:
            # d = v.flatten() - gmse[n].flatten()
            # distance.append(np.median(d.cpu().numpy()))
            d = torch.nn.functional.cosine_similarity(v.flatten(1), gmse[n].flatten(1), -1)
            distance.append(d.cpu().numpy().mean(0))

    par_distances.append(np.asarray(distance))

all_statistics = np.asarray(all_statistics)
par_distances = np.asarray(par_distances).T

fig, ax = plt.subplots()
im = ax.imshow(par_distances)
# plt.colorbar(ax=ax)
cbar = ax.figure.colorbar(im, ax=ax, location='top',
                          label='')
# cbar.ax.set_ylabel(rotation=-90, va="bottom")

ax.set_xlabel('Epochs')
ax.set_xticks([])

ax.set_ylabel('Layer')
ax.set_yticks([])

# plt.show()

fig.savefig(
    "der_gradiewnts_heatmap.pdf",
    bbox_inches='tight',
)