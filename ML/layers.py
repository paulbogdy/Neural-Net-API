import numpy as np
from ML.initializers import *
from ML.activations import *


class BaseLayer:
    """
    This is the base layer, from which each layer inherits

    Each layer is a function that takes an input as the values outputted from the last layer
    and returns values in this layer after calculations
    """
    def __init__(self, trainable=False, **kwargs):
        self.trainable = trainable

    def __call__(self, inputs, **kwargs):
        pass

    def train(self, **kwargs):
        pass


class Flatten(BaseLayer):
    """
    A flatten layer is pretty much an input layer, that has a specific size

    The function returns the input but reshaped to conform to the data
    """
    def __init__(self, size, trainable=False, data_format=None, **kwargs):
        super(Flatten, self).__init__(trainable)
        self.__size = size
        self.activated = None

    def __len__(self):
        return self.__size

    def __call__(self, inputs, batch_size=1, **kwargs):
        self.activated = np.array(inputs.reshape(batch_size, inputs.size//batch_size)).astype(np.float32)
        return self.activated

    def build(self, optimizer, param):
        pass


class Dense(BaseLayer):
    """
    A dense layer is a trainable layer that has weights from each neuron from the layer,
    to each neuron from the last layer

    Attributes
    __________

    biases : the list of biases
    weights : the list of weights, as a matrix
    activation : the activation function, it can be none
    optimizer : the optimizer from the optimizers module
    bias_init : the initializer for biases
    weight_init : the initializer for weights
    size : the size of the layer, integer
    """
    def __init__(self, size, trainable=True,
                 activation=None,
                 weight_init=RandomUniform(),
                 bias_init=Zeroes(),
                 optimizer=None,
                 **kwargs):
        super(Dense, self).__init__(trainable)
        self.__biases = bias_init(size)
        self.__activation = activation
        self.__optimizer = optimizer
        self.activated = None
        self.derived = None
        self.__weight_init = weight_init
        self.__size = size

    def build(self, optimizer, last_size):
        """
        It builds the layer with a specific optimizer
        We also need to know the last layer size in order to compute the dense matrix
        :param optimizer: an optimizer from the optimizers module
        :param last_size: the size of the last layer
        """
        self.__optimizer = optimizer
        self.__weights = self.__weight_init((last_size, self.__size))
        self.__optimizer.fit(self.__weights)

    def __call__(self, inputs, **kwargs):
        """
        The call function returns the basic a(in.dot(w) + b)
        where a is the activation function
        in is the input values
        w is the weight matrix
        b is the biases
        """
        if self.__activation is None:
            self.activated = inputs.dot(self.__weights) + self.__biases
        else:
            self.activated, self.derived = self.__activation(inputs.dot(self.__weights) + self.__biases)
        return self.activated

    def __len__(self):
        return self.__size

    def __compute_gradient(self, inputs, delta):
        """
        Here we compute the gradient from the last gradient
        Each gradient means how much the values of the cost modify when the values in this layer modify
        :param inputs: the input data
        :param delta: the last gradient
        :return: the gradient of biases, of weights and of delta
        """
        gradient = delta
        if self.__activation is not None:
            gradient *= self.derived
        return np.einsum('ij->j', gradient), np.einsum('ij,ik->jk', inputs, gradient), gradient.dot(self.__weights.T)

    def train(self, inputs, delta):
        """
        It optimizes the parameters of the layer in order to minimize the loss function
        :param inputs: the input data
        :param delta: the last gradient
        :return: the gradient of delta, this is used for the next layer and so on
        """
        bias_modifier, weight_modifier, new_delta = self.__compute_gradient(inputs, delta)
        self.__optimizer.optimize(self.__weights, self.__biases,
                                  weights_modifier=weight_modifier, biases_modifier=bias_modifier)
        return new_delta


class Dropout(Dense):
    """
    A dropout is pretty much a dense layer, the only difference being that
    it randomly deactivates some neurons in order to solve over fitting
    """
    def __init__(self, size, rate, trainable=True,
                 activation=None,
                 weight_init=RandomUniform(),
                 bias_init=Zeroes(),
                 optimizer=None,
                 **kwargs):
        super(Dropout, self).__init__(size, trainable, activation, weight_init, bias_init, optimizer)
        self.__rate = rate
        self.__do_dropout = True

    def set_dropout(self, state):
        self.__do_dropout = state

    def __call__(self, inputs, **kwargs):
        super(Dropout, self).__call__(inputs)
        if self.__do_dropout:
            x = np.random.binomial(np.ones(self.activated.shape).astype(np.int32), 1-self.__rate).astype(np.float32)
            self.activated *= x
            self.derived *= x
        return self.activated
