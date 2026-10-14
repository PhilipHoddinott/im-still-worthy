import numpy as np


def rowToImage(pixelRow):
    pixelRow = np.asarray(pixelRow)
    if pixelRow.shape not in ((784,), (1, 784)) or pixelRow.dtype.kind not in 'biufc':
        raise ValueError('pixelRow must be a numeric or logical row with 784 pixels.')
    return pixelRow.reshape((28, 28), order='F')
