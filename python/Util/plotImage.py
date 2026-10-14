import numpy as np
import matplotlib.pyplot as plt
from .NN_forward import NN_forward
from .rowToImage import rowToImage


def plotImage(model, xFit, imageRow=0):
    pixelRow = xFit[imageRow:imageRow + 1, :]
    imageData = rowToImage(pixelRow)
    xArr, yArr = np.meshgrid(np.arange(1, 29), np.arange(1, 29))
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.plot_surface(xArr, yArr, imageData)
    b_cache = NN_forward(model, pixelRow)
    predictedDigit = int(np.argmax(b_cache.prob, axis=1)[0])
    confidence = np.max(b_cache.prob, axis=1)[0]
    axes.set_title(f'Predicted digit: {predictedDigit} ({100 * confidence:.1f}%)')
    plt.show()
