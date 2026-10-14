import numpy as np


def NN_cross_entropy(prob, labels):
    labels = np.asarray(labels).reshape(-1).astype(np.intp)
    numberOfRows = labels.size
    correctProb = prob[np.arange(numberOfRows), labels]
    return -np.mean(np.log(np.maximum(correctProb, 1e-8)))
