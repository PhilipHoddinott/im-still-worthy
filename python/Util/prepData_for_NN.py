from pathlib import Path
from scipy.io import loadmat
from .make_strat_split import make_strat_split


DEFAULT_DATA_FOLDER = Path(__file__).resolve().parents[2] / 'MATLAB' / 'Data'


def prepData_for_NN(dataFolder=DEFAULT_DATA_FOLDER, seed=69):
    dataFolder = Path(dataFolder)
    train = loadmat(dataFolder / 'mnist_training.mat', variable_names=['labels', 'pixel_values'])
    labels = train['labels'].reshape(-1)
    pixel_values = train['pixel_values']
    validFraction = 10 / 60
    fitIdx, validIdx = make_strat_split(labels, validFraction, seed)
    xFit = pixel_values[fitIdx, :]
    yFit = labels[fitIdx]
    xValidation = pixel_values[validIdx, :]
    yValidation = labels[validIdx]
    test = loadmat(dataFolder / 'mnist_test.mat', variable_names=['labels', 'pixel_values'])
    xTest = test['pixel_values']
    yTest = test['labels'].reshape(-1)
    return xFit, yFit, xValidation, yValidation, xTest, yTest
