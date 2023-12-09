import os
import pickle
from collections import defaultdict

import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar = None
    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

acc_results = defaultdict(dict)
bwt_results = defaultdict(dict)

import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("jary-pomponi-r/margin_cl",
                filters={"tags": "margin_simgoid_ablation"})

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    acc_results[run.config['head']['a']][run.config['head']['b']] = run.summary['Top1_Acc_Stream/eval_phase/test_stream/Task000']
    bwt_results[run.config['head']['a']][run.config['head']['b']] = run.summary['StreamBWT/eval_phase/test_stream']

a_vals = sorted(acc_results.keys(), reverse=False)
b_vals = sorted(acc_results[a_vals[0]].keys(), reverse=False)

acc_matrix = np.zeros((len(a_vals), len(b_vals)))
bwt_matrix = np.zeros((len(a_vals), len(b_vals)))

for i, m in enumerate(a_vals):
    acc_matrix[i] = [acc_results[m][r] for r in b_vals]
    bwt_matrix[i] = [bwt_results[m][r] for r in b_vals]

fig, ax = plt.subplots()

im, cbar = heatmap(acc_matrix * 100, a_vals, b_vals, ax=ax,
                   cmap="Purples", cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\beta$')

fig.tight_layout()
# plt.show()
fig.savefig('sigmoid_acc_heatmap.pdf',  bbox_inches='tight')

fig, ax = plt.subplots()

im, cbar = heatmap(bwt_matrix * 100, a_vals, b_vals, ax=ax,
                   cmap="Purples_r", cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.1f}", textcolors=("white", "black"))

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\beta$')

fig.tight_layout()
fig.savefig('sigmoid_bwt_heatmap.pdf',  bbox_inches='tight')

# plt.show()
