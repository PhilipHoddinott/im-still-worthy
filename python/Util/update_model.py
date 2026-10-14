import numpy as np


def update_model(model, gradiants, lr, momentum, velocity):
    for name in ('W1', 'b1', 'W2', 'b2'):
        v = momentum * getattr(velocity, name) - lr * getattr(gradiants, name)
        setattr(velocity, name, v)
        weights = getattr(model, name)
        # MATLAB arithmetic with a single array produces a single result.
        setattr(model, name, (weights + v).astype(weights.dtype))
    return model, velocity
