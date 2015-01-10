__author__ = '1'

from sklearn.neighbors import KernelDensity
import numpy as np

#all parameters except data are additional
#fraction is supposed fraction of outliers
#kernel in (‘gaussian’;’tophat’;’epanechnikov’;’exponential’;’linear’;’cosine’)
#returning values is numbers of rows in the dataset, which are marked as anomalies by algorithm
def kerneldensity(data, fraction=0.02, kernel='gaussian', bandwidth = 1.0):
    numeration = [[i] for i in xrange(1, len(data)+1, 1)]
    numeration = np.array(numeration)
    y = np.hstack((numeration, data))

    kde = KernelDensity(kernel = kernel, bandwidth=bandwidth)
    kde.fit(data)
    score = np.exp(kde.score_samples(data))
    score = score.reshape(numeration.shape)

    y = np.hstack((y, score))
    size = y.shape[1]
    y = tuple(map(tuple, y))

    y = sorted(y, key = lambda x: x[size - 1], reverse=True)

    startindex = int(-round((fraction*len(y)), ndigits=0))-1
    y = y[startindex:]

    y = sorted(y, key = lambda x: x[0])
    y = np.array(y)

    anomalies = y[:,0]
    return anomalies
