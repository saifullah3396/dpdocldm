from typing import List, Optional

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def get_matplotlib_grid(rows, cols, figsize=16):
    ratio = cols / rows
    figsize = (figsize * ratio, figsize)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(
        wspace=0.0,
        hspace=0.0,
        top=1.0 - 0.5 / (rows + 1),
        bottom=0.5 / (rows + 1),
        left=0.5 / (cols + 1),
        right=1 - 0.5 / (cols + 1),
    )
    return fig, gs


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
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


def plot_confusion_matrix(
    cmat: np.array,
    labels: List[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    dpi: int = 70,
):
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    plt.ylabel(y_label, fontsize=16)
    plt.xlabel(x_label, fontsize=16)

    if labels is None:
        labels = range(cmat.shape[0])

    ax = fig.gca()
    im, cb = heatmap(cmat, labels, labels, ax=ax, cmap="Blues")
    ax.spines[:].set_visible(False)
    cb.outline.set_visible(False)
    _ = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.close()
    return image
