import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from dist_functions import calc_distance


def format_data_regression(dist, response, odor_1st, odor_2nd, conc_1st, conc_2nd, delay=None):
    '''
    prepare input data for logistic regression
    :param mouse_id: (n_trials, ) mouse id of each trial in data
    :param response: (n_trials, ) behavior response (go(1) or nogo(0)) of individual trial
    :param odor_1st: (n_trials, ) odor identity of 1st odor presentation
    :param odor_2nd: (n_trials, ) odor identity of 2nd odor presentation
    :param conc_1st: (n_trials, ) indicator variable of 1st odor concentration (0 for low, 1 for high conc)
    :param conc_2nd: (n_trials, ) indicator variable of 2nd odor concentration (0 for low, 1 for high conc)
    :return:
    '''
    num_odors = dist.shape[0]
    num_trials = len(response)
    odor_dist = np.array([dist[odor_1st[i], odor_2nd[i]] for i in range(num_trials)])

    # independent variable representing 1st vs 2nd
    trial_half = [0 if i < len(response) / 2 else 1 for i in range(num_trials)]

    # independent variable for sequence order
    seq_matrix = np.zeros((num_odors, num_odors))
    num_nonmatch = int(num_odors * (num_odors - 1) / 2)  # number of non-match odor pairs
    var_order = np.random.RandomState(seed=0).permutation(np.repeat([-1, 1], int(num_nonmatch / 2)))
    k = 0
    for i in range(num_odors - 1):
        for j in range(i + 1, num_odors):
            seq_matrix[i, j] = var_order[k]
            seq_matrix[j, i] = - var_order[k]
            k += 1

    seq = np.array([seq_matrix[odor_1st[i], odor_2nd[i]] for i in np.arange(num_trials)])

    # dataframe of independent variables
    if delay is None:
        X = pd.DataFrame({'odor': odor_dist, 'conc_1st': conc_1st, 'conc_2nd': conc_2nd, 'trials': trial_half, 'seq': seq})
    else:
        X = pd.DataFrame({'odor': odor_dist, 'conc_1st': conc_1st, 'conc_2nd': conc_2nd, 'trials': trial_half, 'seq': seq, 'delay':delay})

    X = (X - X.mean()) / X.std()

    y = 1 - response
    return X, y


def regression_all_mice(mouse_id, response, odor_1st, odor_2nd, conc_1st, conc_2nd):
    '''
    perform logisitc regression for each mouse separately
    :param mouse_id: (n_trials, ) mouse id of each trial in data
    :param response: (n_trials, ) behavior response (go(1) or nogo(0)) of individual trial
    :param odor_1st: (n_trials, ) odor identity of 1st odor presentation
    :param odor_2nd: (n_trials, ) odor identity of 2nd odor presentation
    :param conc_1st: (n_trials, ) indicator variable of 1st odor concentration (0 for low, 1 for high conc)
    :param conc_2nd: (n_trials, ) indicator variable of 2nd odor concentration (0 for low, 1 for high conc)
    :return:
    '''
    n_mouse = len(np.unique(mouse_id))

    coef_all = []
    bias_all = []
    for m in range(n_mouse):
        idx = np.where(mouse_id == m)[0]
        dist = calc_distance(response[idx], np.c_[odor_1st[idx], odor_2nd[idx]], pool2d=True, return_p_nogo=False)
        X, y = format_data_regression(dist, response[idx], odor_1st[idx], odor_2nd[idx], conc_1st[idx], conc_2nd[idx])

        model = LogisticRegression()
        model.fit(X, y)

        coef_all.append(model.coef_[0])
        bias_all.append(model.intercept_[0])
    return np.array(coef_all), np.array(bias_all)


def regression_all_mice_delay(mouse_id, response, odor_1st, odor_2nd, conc_1st, conc_2nd, delay):
    '''
    perform logisitc regression for each mouse separately
    :param mouse_id: (n_trials, ) mouse id of each trial in data
    :param response: (n_trials, ) behavior response (go(1) or nogo(0)) of individual trial
    :param odor_1st: (n_trials, ) odor identity of 1st odor presentation
    :param odor_2nd: (n_trials, ) odor identity of 2nd odor presentation
    :param conc_1st: (n_trials, ) indicator variable of 1st odor concentration (0 for low, 1 for high conc)
    :param conc_2nd: (n_trials, ) indicator variable of 2nd odor concentration (0 for low, 1 for high conc)
    :param delay: (n_trials, ) indicator variable of delay (0 for short, 1 for long delay)
    :return:
    '''
    n_mouse = len(np.unique(mouse_id))

    coef_all = []
    bias_all = []
    for m in range(n_mouse):
        idx = np.where(mouse_id == m)[0]
        dist = calc_distance(response[idx], np.c_[odor_1st[idx], odor_2nd[idx]], pool2d=True, return_p_nogo=False)
        X, y = format_data_regression(dist, response[idx], odor_1st[idx], odor_2nd[idx], conc_1st[idx], conc_2nd[idx], delay=delay[idx])

        model = LogisticRegression()
        model.fit(X, y)

        coef_all.append(model.coef_[0])
        bias_all.append(model.intercept_[0])
    return np.array(coef_all), np.array(bias_all)


def plot_regression_coefficients(coef, regressors):
    '''
    plot function
    :param coef:
    :param regressors:
    :return:
    '''
    data_points = np.abs(coef)
    data_mean = np.mean(data_points, axis=0)
    data_sem = np.std(data_points, axis=0) / np.sqrt(data_points.shape[0])
    num_mice = data_points.shape[0]

    fig = plt.figure(figsize=(0.7 * len(regressors), 3.5))
    grid = plt.GridSpec(3, 4, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(grid[:-1, 1:])
    # Individual data with dots
    np.random.seed(0)
    for j in np.arange(len(regressors)):
        ax.scatter(j + 0.1 * (np.random.random(num_mice) - 0.5),
                   data_points[:, j], color=[0.9, 0.9, 0.9], s=30, edgecolor=[0.5, 0.5, 0.5], zorder=-1)

    # average across mice
    for j in np.arange(len(regressors)):
        ax.scatter(j, data_mean[j], s=200, zorder=1, marker='_', color='k')
        ax.errorbar(j, data_mean[j], yerr=data_sem[j], zorder=0, ls='none', elinewidth=3, ecolor='k')

    ax.set_xticks(np.arange(len(regressors)))
    ax.set_xticklabels(regressors, rotation=60)

    ax.set_yticks(np.arange(0, 1.7, 0.5))
    ax.set_ylim((0, 1.7))
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((-0.5, len(regressors) + 0.5))

    # fig.tight_layout()
    ax.set_ylabel('|Coefficient|', fontsize=12)
    ax.tick_params(direction='out', bottom='off', left='off', labelleft='on', labelbottom='on')
    return fig


if __name__ == '__main__':

    '''
    Load data
    '''
    path_data = r'.'

    with open(os.path.join(path_data, 'data_odorset1.pkl'), 'rb') as f:
        data1 = pickle.load(f)

    with open(os.path.join(path_data, 'data_odorset2.pkl'), 'rb') as f:
        data2 = pickle.load(f)

    mouse_id_1 = data1['mouse_id']
    response_1 = data1['response']
    odor_1st_1 = data1['odor_1st'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_2nd_1 = data1['odor_2nd'] - 1 # data file contains odor id from 1-24, make it 0-23
    odor_pair_1 = np.c_[odor_1st_1, odor_2nd_1]
    conc_1st_1 = data1['conc_1st']
    conc_2nd_1 = data1['conc_2nd']
    odor_names_1 = data1['odor_names']

    mouse_id_2 = data2['mouse_id']
    response_2 = data2['response']
    odor_1st_2 = data2['odor_1st'] - 1  # data file contains odor id from 1-24, make it 0-23
    odor_2nd_2 = data2['odor_2nd'] - 1  # data file contains odor id from 1-24, make it 0-23
    odor_pair_2 = np.c_[odor_1st_2, odor_2nd_2]
    conc_1st_2 = data2['conc_1st']
    conc_2nd_2 = data2['conc_2nd']
    delay_2 = np.array([0 if d==3000 else 1 for d in data2['delay']])
    odor_names_2 = data2['odor_names']

    '''
    Logistic regression
    '''
    # without delay term
    coef_all, bias_all = regression_all_mice(mouse_id_1, response_1, odor_1st_1, odor_2nd_1, conc_1st_1, conc_2nd_1)

    fig = plot_regression_coefficients(coef_all, ['odor', 'conc_1st', 'conc_2nd', 'trials', 'seq'])
    plt.show()

    # with delay term
    coef_all_delay, bias_all_delay = regression_all_mice_delay(mouse_id_2, response_2, odor_1st_2, odor_2nd_2, conc_1st_2, conc_2nd_2, delay_2)

    fig = plot_regression_coefficients(coef_all_delay, ['odor', 'conc_1st', 'conc_2nd', 'trials', 'seq', 'delay'])
    plt.show()




