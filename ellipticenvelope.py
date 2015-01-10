__author__ = '1'

from sklearn.covariance import EllipticEnvelope
import numpy as np
#all parameters except data are additional
#fraction is an amount of contamination of the data set, i.e. the proportion of outliers in the data set.
#returning values is numbers of rows in the dataset, which are marked as anomalies by algorithm
def ellipticenvelope(data, fraction = 0.02):
    elenv = EllipticEnvelope(contamination=fraction)
    elenv.fit(data)
    score = elenv.predict(data)

    numeration = [[i] for i in xrange(1, len(data)+1, 1)]
    numeration = np.array(numeration)
    y = np.hstack((numeration, score))

    anomalies = numeration
    for num,s in y:
        if (y == 1):
            y = np.delete(anomalies, num-1, axis=0)

    return anomalies

