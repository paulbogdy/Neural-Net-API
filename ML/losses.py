import numpy as np


class Loss:
    def __call__(self, output, expected_output, **kwargs):
        return 0, output


class MeanSquaredError(Loss):

    def __call__(self, output, expected_output, **kwargs):
        return 0.5*np.sum((output - expected_output)**2), output - expected_output


class MeanAbsoluteError(Loss):
    def __call__(self, output, expected_output, **kwargs):
        derivative = np.greater(output, expected_output)
        derivative = 2*derivative.astype(np.float32) - 1
        return np.sum(abs(output - expected_output)), derivative
