import numpy as np


class SGD:
    """
    Sochastic Gradient Descent optimizer

    Attributes
    __________

    learning_rate : float, how fast the learning should happen
        the greater the learning rate the easier to overshoot the result
        the smaller the learning rate the harder to converge

    momentum : float, we use momentum in order to take into consideration the last results
    nestrov : boolean, if true we implement nestrov momentum
    velocity : unique for each layer is the shape of the weights, and here we store last results

    Methods
    _______

    fit : it just fits the optimizer for a specific layer
    optimize : it optimizes the given weights of a layer so that it minimizes the loss value
    """
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, **kwargs):
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__nesterov = nesterov

    def fit(self, weights):
        """
        sets the velocity matrix with the same shape as the weights
        :param weights: the weights of the layer, a matrix, np array
        :return:
        """
        if self.__momentum > 0:
            self.__velocity = np.zeros(shape=weights.shape).astype(np.float32)

    def optimize(self, weights, biases, weights_modifier, biases_modifier, **kwargs):
        """
        It optimizes the weights and biases in order to minimize the loss function
        Depending on momentum and nestrov variables it may behave differently

        with 0 momentum it just subtracts the gradients * learning_rate
        with momentum it updates the velocity and adds the velocity
        with nestrov it adds the velocity*momentum and subtract the gradient*learning_rate

        :param weights: the weights of the layer
        :param biases: the biases of the layer
        :param weights_modifier: the gradient of the weights
        :param biases_modifier: the gradient of the biases
        """
        biases -= self.__learning_rate * biases_modifier
        if self.__momentum == 0:
            weights -= self.__learning_rate * weights_modifier
        elif self.__nesterov:
            self.__velocity = self.__momentum * self.__velocity - self.__learning_rate * weights_modifier
            weights += self.__momentum * self.__velocity - self.__learning_rate * weights_modifier
        else:
            self.__velocity = self.__momentum * self.__velocity - self.__learning_rate * weights_modifier
            weights += self.__velocity
