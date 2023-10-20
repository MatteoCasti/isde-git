import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.metrics import pairwise_distances


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass
def load_mnist_data(filename, n_samples=None):
    """This function returns MNIST handwritten digits and labels as ndarrays."""
    data = pd.read_csv(filename)
    data = np.array(data)  # cast pandas dataframe to numpy array
    if n_samples is not None:  # only returning the first n_samples
        data = data[:n_samples, :]
    y = data[:,0]
    x = data[:,1:] / 255.0
    return x, y


def predict_for_loops(xts, centroids):
    n_samples = xts.shape[0]
    n_classes = centroids.shape[0]
    dist = np.zeros(shape=(n_samples, n_classes))
    ypred = np.zeros(shape=(n_samples,), dtype='int')

    for i in range(n_samples):
        for k in range(n_classes):
            dist[i, k] = np.linalg.norm(xts[i, :] - centroids[k, :], ord=2)
        ypred[i] = np.argmin(dist[i, :])

    return ypred


def predict(xts, centroids):
    dist = pairwise_distances(xts, centroids)  #  dist.shape = (400, 10)
    ypred = np.argmin(dist, axis=1)
    return ypred