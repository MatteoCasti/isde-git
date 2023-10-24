import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.metrics import pairwise_distances



def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass
def load_data(filename, n_samples=None):
    """This function returns MNIST handwritten digits and labels as ndarrays."""
    data = pd.read_csv(filename)
    data = np.array(data)  # cast pandas dataframe to numpy array
    if n_samples is not None:  # only returning the first n_samples
        data = data[:n_samples, :]
    y = data[:,0]
    x = data[:,1:] / 255.0
    return x, y


