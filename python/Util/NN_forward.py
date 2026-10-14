from types import SimpleNamespace
import numpy as np
from .softMax import softMax


def NN_forward(model, XBatch):
    hiddenScore = XBatch @ model.W1 + model.b1
    hiddenVal = np.maximum(0, hiddenScore)
    scores = hiddenVal @ model.W2 + model.b2
    prob = softMax(scores)
    return SimpleNamespace(X=XBatch, hiddenScore=hiddenScore,
                           hiddenValues=hiddenVal, scores=scores, prob=prob)
