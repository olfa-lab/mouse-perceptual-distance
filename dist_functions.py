
import numpy as np

def split_trials_by_mouse(trials, mouse_id):
    '''

    :param trials:
    :param mouse_id:
    :return:
    '''
    trial_numbers = [sum(mouse_id==i) for i in np.unique(mouse_id)]
    trials_by_mouse= np.split(trials, np.cumsum(trial_numbers)[:-1], axis=0)
    return trials_by_mouse

def count_nogo_by_trial_type(response, trial_type):
    """
    count number of nogo responses and total number of trials for each odor pair
    :param response: (n_trials, ) behavior response (go(1) or nogo(0)) of individual trial
    :param odorpair: odors presented in each trial. odorpair[tr, 0] represents 1st odor presentation of trial (tr)
    :param num_odors: number of different odors used in the session
    :return:
    [num_odor, num_odor, 0]: count of nogo trials of each odor pair
    [num_odor, num_odor, 1]: count of total number of trials of each odor pair
    """
    num_conditions = len(np.unique(trial_type))
    ng_count = np.zeros((num_conditions, 2))
    for i in range(num_conditions):
            ng_count[i,  0] = np.sum(1 - response[trial_type==i])  # Count for number of ngo
            ng_count[i,  1] = np.sum(trial_type==i)
    return ng_count


def calc_p_nogo_by_trial_type(response, trial_type):
    '''

    :param response: behavior response (go(1) or nogo(0)) of individual trial
    :param trial_type:
    :return:
    '''
    ng_count = count_nogo_by_trial_type(response, trial_type)
    p_nogo = ng_count[:, 0] / ng_count[:, 1]
    return p_nogo


def count_nogo(response, odorpair, num_odors):
    """
    count number of nogo responses and total number of trials for each odor pair
    :param response: (n_trials, ) behavior response (go(1) or nogo(0)) of individual trial
    :param odorpair: odors presented in each trial. odorpair[tr, 0] represents 1st odor presentation of trial (tr)
    :param num_odors: number of different odors used in the session
    :return:
    [num_odor, num_odor, 0]: count of nogo trials of each odor pair
    [num_odor, num_odor, 1]: count of total number of trials of each odor pair
    """
    ng_count = np.zeros((num_odors, num_odors, 2))
    for i in range(num_odors):
        for j in range(num_odors):
            ng_count[i, j, 0] = np.sum(1 - response[(odorpair[:, 0] == i) & (odorpair[:, 1] == j)])  # Count for number of ngo
            ng_count[i, j, 1] = np.sum((odorpair[:, 0] == i) & (odorpair[:, 1] == j))
    return ng_count


def dist_transform_func(p_nogo):
    """
    Function to convert matrix of p_nogo choice into distance matrix
    :param p_nogo:
    :return: distance metric normalized by diagonal element of p_nogo
    """
    sim = 1 - p_nogo
    sim2 = np.copy(sim)

    for i in range(sim.shape[0]):
        sim2[i, :] = sim2[i, :] / np.sqrt(sim[i, i])
        sim2[:, i] = sim2[:, i] / np.sqrt(sim[i, i])

    dist = 1 - sim2
    dist[dist < 0] = 0.001
    dist[np.diag_indices_from(sim2)] = 0
    return dist


def calc_distance(response, odorpair, num_odors=None, pool2d=True, return_p_nogo=False):
    """
    :param response: (n_trials, ) behavior response (go(1) or nogo(0)) of individual trial
    :param odorpair: odors presented in each trial. odorpair[tr, 0] represents 1st odor presentation of trial (tr)
    :param num_odors: number of different odors used in the session
    :param pool2d: True if you don't discriminate AB vs BA, False otherwise
    :return: perceptual distance between odors
    """
    if num_odors is None:
        num_odors = np.max(odorpair) + 1

    count_all = count_nogo(response, odorpair, num_odors)
    if pool2d and np.ndim(odorpair) == 2:
        count_all = (count_all + np.transpose(count_all, (1, 0, 2))) / 2

    p_nogo = count_all[:, :, 0] / count_all[:, :, 1]
    if not return_p_nogo:
        return dist_transform_func(p_nogo)
    else:
        return dist_transform_func(p_nogo), p_nogo