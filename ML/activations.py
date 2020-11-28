import numpy as np


"""
Here are all the activation functions for layers
"""


class Activation:
    """
    Base activation class, all activations inherit from this one
    Each Activation is a function that takes a parameter and returns the value after mapping the function
    over each element and it's derivative
    """
    def __call__(self, x, **kwargs):
        pass


class Sigmoid(Activation):
    """
    Sigmoid function 1/(1+e^-x)
    """
    def __call__(self, x, **kwargs):
        s = 1.0 / (1.0 + np.exp(-x))
        ds = s * (1 - s)
        return s, ds


class Relu(Activation):
    """
    Relu function 0 if x<0 else x
    """
    def __call__(self, x, **kwargs):
        t = np.copy(x)
        t[t <= 0] = 0
        dt = np.copy(x)
        dt[dt <= 0] = 0
        dt[dt > 0] = 1
        return t, dt


class LeakyRelu(Activation):
    """
    Leaky Relu function 0.01 if x<0 else x
    """
    def __call__(self, x, **kwargs):
        t = np.copy(x)
        t[t <= 0] *= 0.01
        dt = np.copy(x)
        dt[dt <= 0] = 0.01
        dt[dt > 0] = 1
        return t, dt


class Tanh(Activation):
    """
    Tanh function (e^x - e^-x)/(e^x + e^-x)
    """
    def __call__(self, x, **kwargs):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        dt = 1 - t ** 2
        return t, dt


class Softmax(Activation):
    """
    Not working right yet, needs improvement
    """
    def __call__(self, x, **kwargs):
        s = np.exp(x-np.max(x))
        s = s/np.sum(s)
        return s, s*(1-s)
