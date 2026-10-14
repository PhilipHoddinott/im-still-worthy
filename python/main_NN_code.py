"""Direct translation of MATLAB/main_NN_code.m."""
from time import perf_counter
from pathlib import Path
from types import SimpleNamespace
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
from Util.prepData_for_NN import prepData_for_NN, DEFAULT_DATA_FOLDER
from Util.init_NN import init_NN
from Util.NN_forward import NN_forward
from Util.NN_backward import NN_backward
from Util.NN_cross_entropy import NN_cross_entropy
from Util.update_model import update_model
from Util.getFit import getFit


DEFAULT_MODEL_FILE = Path(__file__).resolve().parent / 'trained_model.mat'


def main(user_options=None, dataFolder=DEFAULT_DATA_FOLDER, modelFile=DEFAULT_MODEL_FILE):
    plt.close('all')
    if user_options is None:
        user_options = SimpleNamespace(
            inputSize=784, hiddenSize=256, outputSize=10,
            learnRate=[0.05, 0.02, 0.01], batchSize=128,
            numEpochs=60, seed=69, momentum=0.9)

    xFit, yFit, xValidation, yValidation, xTest, yTest = prepData_for_NN(dataFolder, user_options.seed)
    model, velocity = init_NN(user_options)
    numberOfFitRows = xFit.shape[0]
    history = SimpleNamespace(epoch=np.arange(1, user_options.numEpochs + 1).reshape(-1, 1))
    for name in ('fitLoss', 'validLoss', 'fitAcc', 'validAcc', 'elapsedSec'):
        setattr(history, name, np.zeros((user_options.numEpochs, 1)))
    trainTimer = perf_counter()
    i_lr = 0
    lrr = np.atleast_1d(user_options.learnRate)
    sz_lr = lrr.size

    for i_epoch in range(1, user_options.numEpochs + 1):
        shuffledRows = np.random.permutation(numberOfFitRows)
        if sz_lr > 1:
            ii = i_epoch / user_options.numEpochs
            if ii > ((i_lr + 1) / sz_lr):
                i_lr += 1
        lr = lrr[i_lr]
        i_batch_out_count = 0
        for iBatch in range(0, numberOfFitRows, user_options.batchSize):
            batchEnd = min(iBatch + user_options.batchSize, numberOfFitRows)
            batchRows = shuffledRows[iBatch:batchEnd]
            XBatch = xFit[batchRows, :]
            yBatch = yFit[batchRows]
            b_cache = NN_forward(model, XBatch)
            gradiants = NN_backward(model, b_cache, yBatch)
            model, velocity = update_model(model, gradiants, lr, user_options.momentum, velocity)
            if i_epoch == 1:
                if i_batch_out_count % 10 == 0:
                    _, fitAcc = getFit(model, xFit, yFit)
                    print(f'Epoch {i_epoch} / {user_options.numEpochs} | batch: {iBatch + 1} / {numberOfFitRows} | in batch Fit {fitAcc:.4f}')
                i_batch_out_count += 1

        fitLoss, fitAcc = getFit(model, xFit, yFit)
        history.fitLoss[i_epoch - 1, 0] = fitLoss
        history.fitAcc[i_epoch - 1, 0] = fitAcc
        validProbs = NN_forward(model, xValidation)
        history.validLoss[i_epoch - 1, 0] = NN_cross_entropy(validProbs.prob, yValidation)
        validPred = np.argmax(validProbs.prob, axis=1)
        history.validAcc[i_epoch - 1, 0] = np.mean(validPred == yValidation)
        history.elapsedSec[i_epoch - 1, 0] = perf_counter() - trainTimer
        print(f'Epoch {i_epoch} / {user_options.numEpochs} | lr: {lr:.2f} | '
              f'Fit loss {fitLoss:.4f} | validation loss {history.validLoss[i_epoch - 1, 0]:.4f} | '
              f'Fit {fitAcc:.4f} | validation {history.validAcc[i_epoch - 1, 0]:.4f}')

    savemat(modelFile, {'model': vars(model)}, do_compression=True)
    print(f'Saved model to {Path(modelFile).resolve()}')

    # Expose the script's workspace for interactive use, including its test data.
    return SimpleNamespace(model=model, velocity=velocity, history=history,
                           user_options=user_options, xFit=xFit, yFit=yFit,
                           xValidation=xValidation, yValidation=yValidation,
                           xTest=xTest, yTest=yTest)


if __name__ == '__main__':
    results = main()
    model = results.model
    velocity = results.velocity
    history = results.history
    user_options = results.user_options
    xFit, yFit = results.xFit, results.yFit
    xValidation, yValidation = results.xValidation, results.yValidation
    xTest, yTest = results.xTest, results.yTest
