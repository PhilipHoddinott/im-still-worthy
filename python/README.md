# Python translation

This mirrors `../MATLAB/main_NN_code.m` and all twelve `Util/*.m` functions,
using NumPy rather than a neural-network framework. Structs become
`SimpleNamespace` objects, so fields still use names such as `model.W1` and
`history.fitLoss`. The architecture, hyperparameters, learning-rate schedule,
first-epoch progress reporting, and momentum equations match the MATLAB source.

Install dependencies and run from this folder:

```powershell
python -m pip install -r requirements.txt
python main_NN_code.py
```

SciPy loads `../MATLAB/Data/mnist_training.mat` and `mnist_test.mat` directly.
The default path is relative to the source files, independent of the current
working directory. Pixel values and dtypes are preserved without additional
normalization. Pass another folder to `main(dataFolder=...)` or
`prepData_for_NN(dataFolder, seed)` if needed.

For interactive use:

```python
from main_NN_code import main
from Util.plotImage import plotImage

results = main()
plotImage(results.model, results.xFit, 0)
```

Python array indices start at zero: `plotImage(..., 0)` selects MATLAB's first
row. Labels remain digits 0 through 9. Labels are one-dimensional arrays;
biases, label counts, and history retain MATLAB's row/column shapes.
`rowToImage` uses column-major reshape to preserve MATLAB image orientation.
The original script loads test data but does not evaluate it; this port does
the same. `main()` returns its model, velocity, history, options, and datasets.
Pass a `SimpleNamespace` containing the same option fields to override defaults.

After training, `main()` saves `trained_model.mat` beside `main_NN_code.py`.
The file contains a scalar MATLAB struct named `model`, with the original
weight, bias, and size fields. Each run replaces this file. Pass
`main(modelFile=...)` to choose another output path. In MATLAB:

```matlab
load('trained_model.mat', 'model');
model.W1
```

Seeds are reproducible within Python, but MATLAB and NumPy generate different
random permutations and normal samples. Training is therefore not bit-for-bit
identical. Floating-point arithmetic also differs between implementations.
