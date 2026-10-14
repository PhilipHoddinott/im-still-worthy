from types import SimpleNamespace
import numpy as np


def init_NN(opts):
    inputSize = opts.inputSize
    hiddenSize = opts.hiddenSize
    outputSize = opts.outputSize
    np.random.seed(opts.seed)
    W1 = np.random.randn(inputSize, hiddenSize).astype(np.float32) * np.float32(np.sqrt(2 / inputSize))
    W2 = np.random.randn(hiddenSize, outputSize).astype(np.float32) * np.float32(np.sqrt(2 / hiddenSize))
    b1 = np.zeros((1, hiddenSize), dtype=np.float32)
    b2 = np.zeros((1, outputSize), dtype=np.float32)
    model = SimpleNamespace(W1=W1, b1=b1, W2=W2, b2=b2,
                            inputSize=inputSize, hiddenSize=hiddenSize, outputSize=outputSize)
    # MATLAB zeros(size(...)) defaults to double, unlike the weights.
    velocity = SimpleNamespace(**{name: np.zeros(getattr(model, name).shape)
                                  for name in ('W1', 'b1', 'W2', 'b2')})
    return model, velocity
