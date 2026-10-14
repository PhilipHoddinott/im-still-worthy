import numpy as np
from .NN_forward import NN_forward
from .NN_cross_entropy import NN_cross_entropy


def getFit(model, XFit, yFit):
    fitProb = NN_forward(model, XFit)
    fitLoss = NN_cross_entropy(fitProb.prob, yFit)
    fitPredict = np.argmax(fitProb.prob, axis=1)
    fitAcc = np.mean(fitPredict == np.asarray(yFit).reshape(-1))
    return fitLoss, fitAcc
