__author__ = '1'

import numpy as np
from sklearn.mixture import GMM

#all parameters except data are additional
#fraction is supposed fraction of outliers in the data
#num_components is number of mixture components
#returning values is numbers of rows in the dataset, which are marked as anomalies by algorithm
def gaussianmixture(data, fraction=0.02, num_components=5):
    numeration = [[i] for i in xrange(1, len(data)+1, 1)]
    numeration = np.array(numeration)

    gmm = GMM(n_components=num_components, covariance_type='full', n_init=5)
    gmm.fit(data)

    score = np.exp(gmm.score(data))
    score = score.reshape(numeration.shape)

    y = np.hstack((numeration, score))
    size = y.shape[1]
    y = tuple(map(tuple, y))

    y = sorted(y, key = lambda x: x[size - 1], reverse=True)

    startindex = int(-round((fraction*len(y)), ndigits=0))-1
    anomalies = y[startindex:]

    anomalies = sorted(anomalies, key = lambda x: x[0])
    anomalies = np.array(anomalies)
    anomalies = anomalies[:,0]

    return anomalies

