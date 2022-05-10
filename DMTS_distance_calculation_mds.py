
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from dist_functions import calc_distance


def plot_heatmap(data, odor_names):
    '''
    :param data:
    :param odor_names:
    :return:
    '''
    fig = plt.figure()
    mask = np.ones_like(data, dtype=bool)
    mask[np.tril_indices_from(mask, k=0)]=False
    h = sns.heatmap(data, mask=mask, vmin=np.min(data), vmax=np.max(data), square=True, cmap='Greys_r', ax=plt.gca(),
                      cbar_kws={'label':'', "shrink": .5}, annot=False, fmt=".0f",
                      annot_kws={'size': 12})
    h.set_xticklabels(odor_names)
    h.set_yticklabels(odor_names)
    plt.setp(h.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(h.get_yticklabels(), rotation=0, ha="right", fontsize=12)
    return fig


def mds_calcEmbedding(dist, nd=3):
    '''
    :param dist: distance matrix (n_odors, n_odors)
    :param nd: number of dimensions used for mds
    :return: mds embedding (n_odors, nd)
    '''
    # Calculate coordinate in embedding space
    mds = MDS(n_components=nd, metric=True, n_init=50, max_iter=3000, random_state=19)
    emb = mds.fit(dist).embedding_
    return emb


def plot_scatter3d(pos, odornames, angle=None):
    '''
    :param pos:
    :param odornames:
    :param angle:
    :return:
    '''
    num_odors = pos.shape[0]
    plot_order = np.concatenate(([2, 3, 4, 6, 7, 5, 0, 1], np.arange(8, num_odors)))
    Names = [odornames[x] if x < 8 else '' for x in range(num_odors)]
    Names[9] = 'Mixture'

    fig = plt.figure(figsize=(5,5))
    grid = plt.GridSpec(6, 6, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(grid[:-1, :-1], projection='3d')

    edges = []
    for i in range(num_odors - 1):
        for j in range(i + 1, num_odors):
            if (odornames[i] in odornames[j]) or (odornames[j] in odornames[i]):
                edges.append((i, j))
    segments = [(pos[s, :3], pos[t, :3]) for s, t in edges]

    cmap = plt.cm.get_cmap('jet')
    colors = [[cmap(i / (8 - 1))] if i < 8 else [[0.7, 0.7, 0.7]] for i in range(num_odors)]
    for i, od in enumerate(plot_order):
        ax.scatter(pos[od, 0], pos[od, 1], pos[od, 2], marker='o', c=colors[i], s=32, label=Names[od], alpha=0.7)

    ax.legend(ncol=1, bbox_to_anchor=(0.95, 0.8),
              handlelength=0.1, handletextpad=0.5, borderaxespad=0.1)

    edge_col = Line3DCollection(segments, lw=0.5, linestyles='dashed', colors=(0.5, 0.5, 0.5))  # for NEC fig
    ax.add_collection3d(edge_col)

    if angle is not None:
        ax.view_init(angle[0], angle[1])
    return fig

if __name__ == '__main__':

    '''
    Load data
    '''

    path_data = r'.'

    with open(os.path.join(path_data, 'data_odorset1.pkl'), 'rb') as f:
        data1 = pickle.load(f)

    mouse_id = data1['mouse_id']
    response = data1['response']
    odor_1st = data1['odor_1st'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_2nd = data1['odor_2nd'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_pair = np.c_[odor_1st, odor_2nd]
    conc_1st = data1['conc_1st']
    conc_2nd = data1['conc_2nd']
    odor_names = data1['odor_names']

    '''
    Calculate P_nogo and distance between odor pairs
    '''
    dist, p_nogo = calc_distance(response, odor_pair, num_odors=None, pool2d=True, return_p_nogo=True)

    plot_heatmap(dist, odor_names)
    plt.show()

    plot_heatmap(p_nogo, odor_names)
    plt.show()


    '''
    MDS
    '''
    embedding = mds_calcEmbedding(dist, nd=3)

    # use pca to calculate optimal angle for display (only for visualization)
    pca = PCA(n_components=3)
    pca.fit(embedding[:8, :])
    pc3 = pca.components_[2]
    elev = 180 * (np.arctan(pc3[2] / np.linalg.norm(pc3[:2])) / np.pi)
    azim = 180 * (np.arctan(pc3[1] / pc3[0]) / np.pi)

    plot_scatter3d(embedding, odornames=odor_names, angle=[-elev, -azim])
    plt.show()









