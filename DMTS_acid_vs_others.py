import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dist_functions import calc_distance

if __name__ == '__main__':
    # Comparison of distance between acid odors vs other distances

    '''
    Load data
    '''
    path_data = r'.'

    with open(os.path.join(path_data, 'data_odorset1.pkl'), 'rb') as f:
        data1 = pickle.load(f)

    with open(os.path.join(path_data, 'data_odorset2.pkl'), 'rb') as f:
        data2 = pickle.load(f)

    with open(os.path.join(path_data, 'data_odorset3.pkl'), 'rb') as f:
        data3 = pickle.load(f)


    response_1 = data1['response']
    odor_1st_1 = data1['odor_1st'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_2nd_1 = data1['odor_2nd'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_pair_1 = np.c_[odor_1st_1, odor_2nd_1]
    odor_names_1 = data1['odor_names']

    response_2 = data2['response']
    odor_1st_2 = data2['odor_1st'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_2nd_2 = data2['odor_2nd'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_pair_2 = np.c_[odor_1st_2, odor_2nd_2]
    odor_names_2 = data2['odor_names']

    response_3 = data3['response']
    odor_1st_3 = data3['odor_1st'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_2nd_3= data3['odor_2nd'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_pair_3 = np.c_[odor_1st_3, odor_2nd_3]
    odor_names_3 = data3['odor_names']

    '''
    Calculate distance between odor pairs
    '''
    dist_1, p_nogo_1 = calc_distance(response_1, odor_pair_1, num_odors=None, pool2d=True, return_p_nogo=True)
    dist_2, p_nogo_2 = calc_distance(response_2, odor_pair_2, num_odors=None, pool2d=True, return_p_nogo=True)
    dist_3, p_nogo_3 = calc_distance(response_3, odor_pair_3, num_odors=None, pool2d=True, return_p_nogo=True)

    # acid: all component odors are acid
    # nonacid: at least one of component odors are not acid
    idx_acid_1 = [sum([str.find(x, 'Acd') > 0 for x in xx]) == len(xx) for xx in [x.split('_') for x in odor_names_1]]
    idx_nonacid_1 = [sum([str.find(x, 'Acd') > 0 for x in xx]) == 0 for xx in [x.split('_') for x in odor_names_1]]
    idx_acid_2 = [sum([str.find(x, 'Acd') > 0 for x in xx]) == len(xx) for xx in [x.split('_') for x in odor_names_2]]
    idx_nonacid_2 = [sum([str.find(x, 'Acd') > 0 for x in xx]) == 0 for xx in [x.split('_') for x in odor_names_2]]
    idx_acid_3 = [sum([str.find(x, 'Acd') > 0 for x in xx]) == len(xx) for xx in [x.split('_') for x in odor_names_3]]
    idx_nonacid_3 = [sum([str.find(x, 'Acd') > 0 for x in xx]) == 0 for xx in [x.split('_') for x in odor_names_3]]

    dist_acid = []
    dist_others = []
    dist_acid.append(dist_1[np.ix_(idx_acid_1,idx_acid_1)][np.tril_indices_from(dist_1[np.ix_(idx_acid_1,idx_acid_1)],k=-1)])
    dist_acid.append(dist_2[np.ix_(idx_acid_2,idx_acid_2)][np.tril_indices_from(dist_2[np.ix_(idx_acid_2,idx_acid_2)],k=-1)])
    dist_acid.append(dist_3[np.ix_(idx_acid_3,idx_acid_3)][np.tril_indices_from(dist_3[np.ix_(idx_acid_3,idx_acid_3)],k=-1)])
    dist_acid = np.concatenate(dist_acid)

    dist_others.append(dist_1[np.ix_(idx_acid_1, idx_nonacid_1)].ravel())
    dist_others.append(dist_2[np.ix_(idx_acid_2, idx_nonacid_2)].ravel())
    dist_others.append(dist_3[np.ix_(idx_acid_3, idx_nonacid_3)].ravel())
    dist_others = np.concatenate(dist_others)

    '''
    plot
    '''
    fig = plt.figure(figsize=(3,3))
    grid = plt.GridSpec(5, 6, hspace=1, wspace=0.1)
    ax = fig.add_subplot(grid[:-1, 1:])
    # ax.hist([dist_acid,dist_others], bins=np.arange(0,1.01,0.05), rwidth=2)
    vp1 = ax.violinplot(dist_acid, positions=[1], showmedians=True, showextrema=True)
    vp2 = ax.violinplot(dist_others, positions=[2], showmedians=True, showextrema=True)

    for partname in ['cbars', 'cmedians']:
        vp = vp1[partname]
        vp.set_edgecolor([0.3, 0.3, 0.3])
        vp.set_linewidth(1)
        vp = vp2[partname]
        vp.set_edgecolor([0.3, 0.3, 0.3])
        vp.set_linewidth(1)

    for partname in ['cmins', 'cmaxes']:
        vp = vp1[partname]
        vp.set_linewidth(0)
        vp = vp2[partname]
        vp.set_linewidth(0)

    for vp in vp1['bodies']:
        vp.set_facecolor([0.7, 0.7, 0.7])
        vp.set_edgecolor([0.7, 0.7, 0.7])
        vp.set_linewidth(0)
        vp.set_alpha(0.5)
    for vp in vp2['bodies']:
        vp.set_facecolor([0.7, 0.7, 0.7])
        vp.set_edgecolor([0.7, 0.7, 0.7])
        vp.set_linewidth(0)
        vp.set_alpha(0.5)

    ax.tick_params(direction='out',bottom='off', left='off',labelleft='on', labelbottom='on')
    ax.set_xticks(np.arange(1,3))
    ax.set_xticklabels([f"acid\nvs\nacid", "acid\nvs\nothers"])
    ax.set_yticks(np.arange(0,1.1,0.25))
    ax.set_yticklabels(['0.0','','0.5','','1.0'])
    ax.set_ylim([0,1])
    ax.set_xlim([0.5,2.5])
    ax.set_ylabel('D')
    plt.show()


