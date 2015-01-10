__author__ = '1'

from sklearn.svm import OneClassSVM
import numpy as np

#all parameters except data are additional
#kernel in  (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’)
#degree is int degree of hte polynomial kernel function and applicable only for 'poly' kernel
#gamma is float kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ kernels
#coeff is independent term in kernel function and applicable in ‘poly’ and ‘sigmoid’ kernels
def svm(data, fraction=0.05, kernel='poly', degree=3, gamma=0, coeff=0):
    svm = OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, nu=fraction, coeff0=coeff)
    svm.fit(data)

    score = svm.predict(data)
    numeration = [[i] for i in xrange(1, len(data)+1, 1)]
    numeration = np.array(numeration)
    y = np.hstack((numeration, score))

    anomalies = numeration
    for num,s in y:
        if (y == 1):
            y = np.delete(anomalies, num-1, axis=0)

    return anomalies
