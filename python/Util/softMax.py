import numpy as np


def softMax(scores):
    shiftedScores = scores - np.max(scores, axis=1, keepdims=True)
    expos = np.exp(shiftedScores)
    return expos / np.sum(expos, axis=1, keepdims=True)
