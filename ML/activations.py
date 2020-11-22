import numpy as np


class Activation:
    def __call__(self, x, **kwargs):
        return 0


class Sigmoid(Activation):
    def __call__(self, x, **kwargs):
        s = 1.0 / (1.0 + np.exp(-x))
        ds = s * (1 - s)
        return s, ds


class Relu(Activation):
    def __call__(self, x, **kwargs):
        t = np.copy(x)
        t[t <= 0] = 0
        dt = np.copy(x)
        dt[dt <= 0] = 0
        dt[dt > 0] = 1
        return t, dt


class LeakyRelu(Activation):
    def __call__(self, x, **kwargs):
        t = np.copy(x)
        t[t <= 0] *= 0.01
        dt = np.copy(x)
        dt[dt <= 0] = 0.01
        dt[dt > 0] = 1
        return t, dt


class Tanh(Activation):
    def __call__(self, x, **kwargs):
        t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        dt = 1 - t ** 2
        return t, dt
