#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_cmaps
import matplotlib.colors as mpl_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_covariance_heatmap(ax, X):
    return plot_heatmap(ax, np.cov(X))

def plot_heatmap(ax, X):
    '''
    Helps to plot a heatmap
    '''
    cmap1 = mpl_cmaps.get_cmap("YlOrRd").copy()
    cmap1.set_bad("w")
    norm1 = mpl_colors.TwoSlopeNorm(vmin=0., vcenter=0.5, vmax=1.)
    im1 = ax.imshow(X.T, cmap = cmap1, norm = norm1, interpolation='nearest', origin = 'upper')

    ax.tick_params(bottom = False, top = True, left = True, right = False,
                    labelbottom = False, labeltop = True, labelleft = True, labelright = False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im1, cax=cax, fraction = 0.1)
    return


def plot_principal_components(pcomp, class_labels, class_props,
        ncomp = 6, 
        subplot_h = 2.0, bgcolor = "#F0F0F0"):

    '''
    pcomp: principal components
    class_labels: the class of each sample
    class_props: list of class properties
    '''

    nrow = ncomp - 1
    ncol = ncomp - 1
    figw = ncol * subplot_h + (ncol - 1) * 0.3 + 1.2
    figh = nrow * subplot_h + (nrow - 1) * 0.3 + 1.5


    def make_plot_principal_components(ax, i, j, comp, labels, unique_labels):
        pc1 = comp[:, j]
        pc2 = comp[:, i]
        for label in unique_labels:
            idx = np.array([k for k, x in enumerate(labels) if x == label])
            ax.scatter(pc1[idx], pc2[idx], s = 50, alpha = 0.7, label = label)
        return

    fig = plt.figure(figsize = (figw, figh))
    axmain = fig.add_subplot(111)
    axs = list()

    for i in range(1, nrow + 1):
        for j in range(ncol):
            ax = fig.add_subplot(nrow, ncol, ((i - 1) * ncol) + j + 1)

            ax.tick_params(bottom = False, top = False, left = False, right = False,
                labelbottom = False, labeltop = False, labelleft = False, labelright = False)
            if j == 0: ax.set_ylabel(f"PC{i + 1}")
            if i == nrow: ax.set_xlabel(f"PC{j + 1}")
            if i > j:
                ax.patch.set_facecolor(bgcolor)
                ax.patch.set_alpha(0.3)
                make_plot_principal_components(ax, i, j, pcomp, class_labels, class_props)
                for side, border in ax.spines.items():
                    border.set_color(bgcolor)
            else:
                ax.patch.set_alpha(0.)
                for side, border in ax.spines.items():
                    border.set_visible(False)

            if i == 1 and j == 0:
                mhandles, mlabels = ax.get_legend_handles_labels()
            axs.append(ax)

    axmain.tick_params(bottom = False, top = False, left = False, right = False,
        labelbottom = False, labeltop = False, labelleft = False, labelright = False)
    for side, border in axmain.spines.items():
        border.set_visible(False)
    axmain.legend(handles = mhandles, labels = mlabels, loc = 'upper right', bbox_to_anchor = (0.9, 0.9))

    plt.tight_layout()
    return axmain, axs
