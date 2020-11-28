import numpy as np


class BaseInitializer:
    def __call__(self, shape, **kwargs):
        return np.empty(shape=shape).astype(np.float32)
    pass


class Zeroes(BaseInitializer):
    def __call__(self, shape, **kwargs):
        return np.zeros(shape=shape).astype(np.float32)


class RandomNormal(BaseInitializer):
    def __call__(self, shape, mean=0.0, stddev=0.05, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(stddev, mean, shape).astype(np.float32)


class RandomUniform(BaseInitializer):
    def __call__(self, shape, low=-0.05, high=0.05, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, shape).astype(np.float32)
