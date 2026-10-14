"""Run with python -m unittest -v test_port from this folder."""
import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from scipy.io import loadmat
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from Util.init_NN import init_NN
from Util.NN_forward import NN_forward
from Util.NN_backward import NN_backward
from Util.NN_cross_entropy import NN_cross_entropy
from Util.make_strat_split import make_strat_split
from Util.prepData_for_NN import prepData_for_NN
from Util.rowToImage import rowToImage
from Util.softMax import softMax
from Util.update_model import update_model
from main_NN_code import main


class PortTests(unittest.TestCase):
    def test_gradients_and_cache_preservation(self):
        opts = SimpleNamespace(inputSize=3, hiddenSize=4, outputSize=2, seed=69)
        model, _ = init_NN(opts)
        for name in ('W1', 'W2', 'b1', 'b2'):
            setattr(model, name, getattr(model, name).astype(np.float64))
        X = np.array([[0.2, 0.7, -0.3], [0.5, -0.1, 0.8]])
        labels = np.array([0, 1])
        cache = NN_forward(model, X)
        original = cache.prob.copy()
        gradients = NN_backward(model, cache, labels)
        np.testing.assert_array_equal(cache.prob, original)
        step = 1e-6
        for name in ('W1', 'W2', 'b1', 'b2'):
            weights = getattr(model, name)
            for index in np.ndindex(weights.shape):
                value = weights[index]
                weights[index] = value + step
                plus = NN_cross_entropy(NN_forward(model, X).prob, labels)
                weights[index] = value - step
                minus = NN_cross_entropy(NN_forward(model, X).prob, labels)
                weights[index] = value
                self.assertAlmostEqual(getattr(gradients, name)[index], (plus - minus) / (2 * step), places=7)

    def test_stratification_rounding_and_seed(self):
        labels = np.repeat(np.arange(10), 3)
        fit, valid = make_strat_split(labels, 0.15, 69)
        self.assertEqual(valid.size, 5)  # round(4.5) in MATLAB
        np.testing.assert_array_equal(np.sort(np.concatenate((fit, valid))), np.arange(30))
        np.testing.assert_array_equal(np.bincount(labels[valid], minlength=10), [1] * 5 + [0] * 5)
        fit2, valid2 = make_strat_split(labels, 0.15, 69)
        np.testing.assert_array_equal(fit, fit2)
        np.testing.assert_array_equal(valid, valid2)

    def test_softmax_and_image_orientation(self):
        probs = softMax(np.array([[10000., 10001., 9999.]]))
        self.assertTrue(np.all(np.isfinite(probs)))
        np.testing.assert_allclose(probs.sum(axis=1), 1)
        image = rowToImage(np.arange(784))
        self.assertEqual(image[1, 0], 1)
        self.assertEqual(image[0, 1], 28)

    def test_momentum_two_steps(self):
        model = SimpleNamespace(**{n: np.ones((1, 1), dtype=np.float32) for n in ('W1', 'W2', 'b1', 'b2')})
        velocity = SimpleNamespace(**{n: np.zeros((1, 1)) for n in vars(model)})
        gradients = SimpleNamespace(**{n: np.full((1, 1), 2.) for n in vars(model)})
        for _ in range(2):
            model, velocity = update_model(model, gradients, 0.1, 0.9, velocity)
        np.testing.assert_allclose(model.W1, 0.42)
        np.testing.assert_allclose(velocity.W1, -0.38)
        self.assertEqual(model.W1.dtype, np.float32)

    def test_real_mat_files_and_training(self):
        data = prepData_for_NN()
        self.assertEqual([a.shape for a in data], [(50000, 784), (50000,), (10000, 784), (10000,), (10000, 784), (10000,)])
        self.assertTrue(np.all(np.isfinite(data[0])))
        small = tuple(a[:256] if i < 2 else a[:64] for i, a in enumerate(data))
        opts = SimpleNamespace(inputSize=784, hiddenSize=16, outputSize=10,
                               learnRate=[0.05, 0.02, 0.01], batchSize=128,
                               numEpochs=3, seed=69, momentum=0.9)
        with TemporaryDirectory() as folder:
            modelFile = Path(folder) / 'trained_model.mat'
            with patch('main_NN_code.prepData_for_NN', return_value=small), contextlib.redirect_stdout(io.StringIO()) as output:
                results = main(opts, modelFile=modelFile)
            savedModel = loadmat(modelFile)['model']
            self.assertEqual(savedModel.shape, (1, 1))
            self.assertEqual(set(savedModel.dtype.names), set(vars(results.model)))
            for name, value in vars(results.model).items():
                np.testing.assert_array_equal(savedModel[name][0, 0], np.atleast_2d(value))
        for rate in ('0.05', '0.02', '0.01'):
            self.assertIn(f'lr: {rate}', output.getvalue())
        self.assertTrue(np.all(np.isfinite(results.history.fitLoss)))
        self.assertLess(results.history.fitLoss[-1, 0], results.history.fitLoss[0, 0])


if __name__ == '__main__':
    unittest.main()
