import numpy as np
import random
from ML.components.functions import *
from ML.handlers.errors import LayerError


class InputLayer:
    def __init__(self, layer_size):
        self.layer_values = np.empty(layer_size)

    def __len__(self):
        return len(self.layer_values)

    def get_layer_values(self):
        return self.layer_values

    def set_layer_values(self, layer_values):
        if len(layer_values) != len(self.layer_values):
            raise LayerError("Can't fit input data on layer!")
        self.layer_values = layer_values


class Layer(InputLayer):
    def __init__(self, last_layer_size, layer_size, activation_function=Identity):
        super().__init__(layer_size)
        #for testing use seed 69
        #np.random.seed(69)
        #random.seed(69)
        self.__weights = np.random.rand(last_layer_size, layer_size)
        self.__biases = np.random.rand(layer_size)
        self.__activation_function = activation_function

    def get_weights(self):
        return self.__weights

    def update_weights(self, modifier):
        self.__weights -= modifier

    def get_biases(self):
        return self.__biases

    def update_biases(self, modifier):
        self.__biases -= modifier

    def get_activation_function(self):
        return self.__activation_function

    def get_layer_values(self):
        return self.layer_values

    def calculate_layer(self, last_layer_values):
        if len(last_layer_values) != len(self.__weights):
            raise LayerError("Impossible Matrix Multiplication!")
        return np.add(self.__biases, np.matmul(last_layer_values, self.__weights))

    def calculate_activated(self, layer_values):
        return self.__activation_function.f(layer_values)

    def calculate_derived(self, layer_values):
        return self.__activation_function.df(layer_values)
