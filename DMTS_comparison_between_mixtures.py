import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from dist_functions import split_trials_by_mouse, calc_p_nogo_by_trial_type


def odor2vec(odor_ids, odornames):
    '''
    convert odor name to binary vector of length equal to number of odors used in experiments
    :param odor_ids: (n_trials, ) string of odornames, 'X' for monomolecular odors, 'X_Y' for mixtures
    :param odornames: List of string containing odor names used in experiments
    :return: odor_vectors (n_trials, n_total_odors), binary matrix about odors presented at each trial
    '''
    n_total_odors = len(odornames)
    n_trials = len(odor_ids)
    odor_vectors = np.zeros((n_trials, n_total_odors))
    for idx, odor_id in enumerate(odor_ids):
        for component in odornames[odor_id].split('_'):
            odor_vectors[idx, odornames.index(component)] = 1
    return odor_vectors


def create_trial_id_by_mixture_types(odor_1st, odor_2nd, odor_names):
    '''
    assign different trial id based on 1st and 2nd odors
    0: 'Match'
    1: 'Non-match (no overlap in compoennts between 1st and 2nd odors)'
    2: 'Non-match (AB vs AC)'
    3: 'Non-match (A vs AB)'
    :param odor_1st:
    :param odor_2nd:
    :param odor_names:
    :return:
    '''

    odorvec_1st = odor2vec(odor_1st, odor_names)
    odorvec_2nd = odor2vec(odor_2nd, odor_names)

    pdist = []
    for i in range(len(odor_1st)):
        pdist.append(cosine(odorvec_1st[i, :], odorvec_2nd[i, :]))

    # a0 = [1 if x <= 1 else 0 for x in range(len(odornames))]
    # a1 = [1 if x == 0 else 0 for x in range(len(odornames))]
    # a2 = [1 if x == 0 or x == 2 else 0 for x in range(len(odornames))]
    # a3 = [1 if x == 2 or x == 3 else 0 for x in range(len(odornames))]
    a0 = [1, 1, 0, 0]
    a1 = [0, 0, 1, 1]
    a2 = [1, 0, 1, 0]
    a3 = [1, 0, 0, 0]
    cosines = [cosine(a0, a0), cosine(a0, a1), cosine(a0, a2), cosine(a0, a3)]

    trial_type = [cosines.index(d) for d in pdist]
    return np.array(trial_type)


if __name__ == '__main__':
    # Comparing p_nogo of odor pairs with different types of mixtures

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
    trial_type = create_trial_id_by_mixture_types(odor_1st, odor_2nd, odor_names)
    response_by_mouse = split_trials_by_mouse(response, mouse_id)
    trial_type_by_mouse = split_trials_by_mouse(trial_type, mouse_id)

    p_nogo_all = calc_p_nogo_by_trial_type(response, trial_type)
    
    p_nogo_by_mouse = [calc_p_nogo_by_trial_type(res, ttype) for res, ttype in zip(response_by_mouse, trial_type_by_mouse)]

    data_sem = np.std(p_nogo_by_mouse, axis=0) / np.sqrt(len(p_nogo_by_mouse))
    num_mice = len(p_nogo_by_mouse)

    '''
    plot
    '''
    fig = plt.figure(figsize=(4,3))
    grid = plt.GridSpec(3, 4, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(grid[:-1, 1:])

    # Individual data with dots
    np.random.seed(0)
    for p in p_nogo_by_mouse:
        ax.scatter(np.arange(4) + 0.1 * (np.random.random(1)), p, color=[0.9, 0.9, 0.9], s=30, edgecolor=[0.5, 0.5, 0.5], zorder=-1)

    # average across mice
    ax.scatter(np.arange(len(p_nogo_all)), p_nogo_all, s=200, zorder=1, marker='_', color='k')
    ax.errorbar(np.arange(len(p_nogo_all)), p_nogo_all, yerr=data_sem, zorder=0, ls='none', elinewidth=3, ecolor='k')

    ax.set_xticks(np.arange(len(p_nogo_all)))
    ax.set_xticklabels(['Match', 'A vs B', 'AC vs BC', 'A vs AB'], rotation=60)

    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_ylim((0, 1))
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((-0.5, len(p_nogo_all) + 0.5))

    # fig.tight_layout()
    ax.set_ylabel('P_nogo', fontsize=12)
    ax.tick_params(direction='out', bottom='off', left='off', labelleft='on', labelbottom='on')
    plt.show()
