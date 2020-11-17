import math
import numpy as np


class Function:
    @staticmethod
    def f(x):
        pass

    @staticmethod
    def df(x):
        pass


class Tanh(Function):
    @staticmethod
    def f(x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    @staticmethod
    def df(x):
        return 1 - Tanh.f(x)**2


class Sigmoid(Function):
    @staticmethod
    def f(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def df(x):
        return Sigmoid.f(x)*(1-Sigmoid.f(x))


class Identity(Function):
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def df(x):
        return 1


class Binomial:
    @staticmethod
    def f(x, y):
        pass

    @staticmethod
    def df(x, y):
        pass


class Loss(Binomial):
    @staticmethod
    def f(x, y):
        return (x-y)**2

    @staticmethod
    def df(x, y):
        return 2*(x-y)
