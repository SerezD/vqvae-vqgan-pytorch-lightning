import torch

from sklearn.manifold import MDS, TSNE

import numpy as np

from matplotlib import pyplot as plt


seed = 4219


def close_all_plots():
    plt.close('all')


def codebook_usage_barplot(y):
    """
    :param y: heights for bar plot, numpy array.
    :return: the bar plot as plt
    """
    f = plt.figure()
    plt.bar(x=np.arange(0, len(y)), height=y)
    plt.xlabel('Codebook Indexes')
    plt.ylabel('Probability of Being chosen')
    plt.grid(True)
    return f


def codebook_multidimensional_scaling(codebook):
    """
    :param codebook: matrix on which mds will be computed
    :return: the 2d projection and distance matrix
    """

    # euclidean distance
    # cast codebook to double since cdist will fail otherwise (bad precision)
    codebook = codebook.double().detach()
    cdist_matrix = torch.cdist(codebook, codebook).cpu().numpy()
    mds = MDS(dissimilarity='precomputed', random_state=seed)
    proj = mds.fit_transform(cdist_matrix)

    # mds plot
    f = plt.figure()
    cols = np.arange(codebook.shape[0])
    plt.scatter(proj[:, 0], proj[:, 1], c=cols, cmap='gist_rainbow')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('codebook index', rotation=270)
    plt.suptitle('Codebook Euclidean Metric Distance Scaling on 2D')

    # euclidean matrix
    p = plt.figure()
    plt.imshow(cdist_matrix, cmap='BuGn')
    plt.colorbar()
    plt.suptitle('Euclidean Distances on Codebook')

    return f, p


def codebook_tsne_proj(codebook):
    """
    :param codebook: torch matrix on which tsne will be performed.
    :return: 2d tsne projection
    """
    codebook = codebook.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=codebook.shape[1] - 10, init='pca', learning_rate=200, random_state=seed)
    proj = tsne.fit_transform(codebook)

    # plot
    f = plt.figure()
    cols = np.arange(codebook.shape[0])
    plt.scatter(proj[:, 0], proj[:, 1], c=cols, cmap='gist_rainbow')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('codebook index', rotation=270)
    plt.suptitle('TSNE display')

    return f
