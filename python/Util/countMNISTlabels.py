import numpy as np


def countMNISTlabels(labels):
    labels = np.asarray(labels).reshape(-1).astype(np.intp)
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError('MNIST labels must be digits 0 through 9.')
    return np.bincount(labels, minlength=10).reshape(10, 1)
