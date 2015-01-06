__author__ = '1'

def gmm(X, anomaly_fraction):
    import numpy as np
    from sklearn.mixture import GMM
    from sklearn.cross_validation import ShuffleSplit
    num_splits = 10
    max_num_components = 10
    curr_total_loss = 0;
    n = 0;
    for num_components in xrange(max_num_components):
        split = ShuffleSplit(n=len(X), n_iter=num_splits, test_size=0.2)
        for train_index, test_index in split:
            gmm = GMM(n_components = num_components + 1, covariance_type = 'full')
            gmm.fit(X[train_index])
            loss = gmm_loss_function(gmm, X, test_index)
            if (num_components == 0) :
                curr_total_loss = loss
                n = num_components + 1
            elif (loss < curr_total_loss):
                curr_total_loss = loss
                n = num_components + 1
end

def gmm_loss_function(gmm, X, test_index):
end