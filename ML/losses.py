import numpy as np


"""
In this module are all the loss function implemented so far
"""


class Loss:
    """
    This is the base loss, all losses inherit from this one
    all losses are functions that take the result and the expected result and calculate a function
    of those two resulting one element
    they also calculate the derivative in order to compute the gradient
    """
    def __call__(self, output, expected_output, **kwargs):
        return 0, output


class MeanSquaredError(Loss):
    """
    Mean squared error (x - y)^2
    """
    def __call__(self, output, expected_output, **kwargs):
        return 0.5*np.sum((output - expected_output)**2), output - expected_output


class MeanAbsoluteError(Loss):
    """
    Mean absolute error abs(x-y)
    """
    def __call__(self, output, expected_output, **kwargs):
        derivative = np.greater(output, expected_output)
        derivative = 2*derivative.astype(np.float32) - 1
        return np.sum(abs(output - expected_output)), derivative
