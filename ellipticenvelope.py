__author__ = '1'

from sklearn.covariance import EllipticEnvelope

#all parameters except data are additional
#fraction is an amount of contamination of the data set, i.e. the proportion of outliers in the data set.
#returning values is numbers of rows in the dataset, which are marked as anomalies by algorithm
def ellipticenvelope(data, fraction = 0.02):
