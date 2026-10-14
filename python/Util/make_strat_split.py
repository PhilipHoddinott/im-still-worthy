import numpy as np
from .countMNISTlabels import countMNISTlabels


def make_strat_split(labels, validFraction, seed):
    labels = np.asarray(labels).reshape(-1)
    np.random.seed(seed)
    labelCounts = countMNISTlabels(labels).reshape(-1)
    nRows = labels.size
    # MATLAB round rounds positive halves up.
    numberofValidRows = int(np.floor(validFraction * nRows + 0.5))
    exactValidationCounts = validFraction * labelCounts
    validationCounts = np.floor(exactValidationCounts).astype(np.intp)
    rowsLeft = numberofValidRows - np.sum(validationCounts)
    allocationorder = np.argsort(-(exactValidationCounts - validationCounts), kind='stable')
    validationCounts[allocationorder[:rowsLeft]] += 1
    fitIndices = []
    validationIndices = []
    for digit in range(10):
        classRows = np.flatnonzero(labels == digit)
        classRows = classRows[np.random.permutation(classRows.size)]
        classValidationCount = validationCounts[digit]
        validationIndices.append(classRows[:classValidationCount])
        fitIndices.append(classRows[classValidationCount:])
    fitIndices = np.concatenate(fitIndices)
    validationIndices = np.concatenate(validationIndices)
    fitIdx = fitIndices[np.random.permutation(fitIndices.size)]
    validIdx = validationIndices[np.random.permutation(validationIndices.size)]
    return fitIdx, validIdx
