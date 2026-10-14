from types import SimpleNamespace
import numpy as np


def NN_backward(model, b_cache, labels):
    labels = np.asarray(labels).reshape(-1).astype(np.intp)
    numberOfRows = labels.size
    # MATLAB assignment copies the array; NumPy needs an explicit copy.
    outGradiant = b_cache.prob.copy()
    outGradiant[np.arange(numberOfRows), labels] -= 1
    outGradiant = outGradiant / numberOfRows
    gradiants = SimpleNamespace()
    gradiants.W2 = b_cache.hiddenValues.T @ outGradiant
    gradiants.b2 = np.sum(outGradiant, axis=0, keepdims=True)
    hiddenValGrad = outGradiant @ model.W2.T
    hiddenSocreGrad = hiddenValGrad * (b_cache.hiddenScore > 0)
    gradiants.W1 = b_cache.X.T @ hiddenSocreGrad
    gradiants.b1 = np.sum(hiddenSocreGrad, axis=0, keepdims=True)
    return gradiants
