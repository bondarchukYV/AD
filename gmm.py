__author__ = '1'

import numpy as np
from sklearn.mixture import GMM
from sklearn.cross_validation import ShuffleSplit

def gmm(data, anomaly_fraction):
    num_splits = 10
    max_num_components = 10
    curr_total_loss = 0;
    num_components = 0;

    for n in xrange(max_num_components):
        gmm = GMM(n_components = n + 1, covariance_type = 'full')
        gmm.fit(data)
        loss = gmm_loss_function(gmm, data)
        if (n == 0) :
            curr_total_loss = loss
            n = n + 1
        elif (loss < curr_total_loss):
            curr_total_loss = loss
            num_components = n + 1

    gmm = GMM(n_components = num_components, covariance_type = 'full')
    gmm.fit(data)
    return get_less_possible(gmm, data, anomaly_fraction)

def gmm_loss_function(gmm, data):
    return log_likelihood(gmm, data)

#def log_likelihood_reg(gmm, data, test_index):

def log_likelihood(gmm, data):
    log_likelihood = 1
    scores = gmm.score(data)
    for data in scores:
        log_likelihood += data
    return log_likelihood

def get_less_possible(gmm, data, fraction):
    probs = gmm.score(data)
    dataprobs = np.hstack(data, probs)
    np.sort(dataprobs, axis = -1, kind = 'mergesort')
    return dataprob[(-len(dataprob)*fraction ):-1, 0:3]