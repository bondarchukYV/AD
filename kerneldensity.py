__author__ = '1'

from sklearn.neighbors import KernelDensity
import numpy as np

def kerneldensity(x, fraction):
def kerneldensity(x, fraction):
    numeration = [[i] for i in xrange(1, len(x)+1, 1)]
    numeration = np.array(numeration)
    y = np.hstack((numeration, x))

    kde = KernelDensity(kernel = 'gaussian', bandwidth=1)
    kde.fit(x)
    score = np.exp(kde.score_samples(x))
    score = score.reshape(numeration.shape)

    y = np.hstack((y, score))
    size = y.shape[1]
    y = tuple(map(tuple, y))

    y = sorted(y, key = lambda x: x[size - 1], reverse=True)

    startindex = int(-round((fraction*len(y)), ndigits=0))-1
    y = y[startindex:]

    y = sorted(y, key = lambda x: x[0])
    y = np.array(y)
    return y[:,0]
