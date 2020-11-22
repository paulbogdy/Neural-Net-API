import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, **kwargs):
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__nesterov = nesterov

    def fit(self, weights):
        if self.__momentum > 0:
            self.__velocity = [np.zeros(weight.shape).astype(np.float32) for weight in weights]

    def optimize(self, weights, biases, weights_modifier, biases_modifier, index, **kwargs):
        if self.__momentum == 0:
            weights -= self.__learning_rate * weights_modifier
            biases -= self.__learning_rate * biases_modifier
        elif self.__nesterov:
            self.__velocity[index] = self.__momentum * self.__velocity[index] - self.__learning_rate * weights_modifier
            weights += self.__momentum * self.__velocity[index] - self.__learning_rate * weights_modifier
        else:
            self.__velocity[index] = self.__momentum * self.__velocity[index] - self.__learning_rate * weights_modifier
            weights += self.__velocity[index]
