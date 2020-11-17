from ML.components.layers import *
from ML.dataManager.data import TrainingData
from ML.handlers.validators import NNValidator
from ML.components.functions import *
import numpy as np


class DeepFeedForward:
    """
    A class used to represent a deep feed forward neural network

    ...

    Attributes
    __________

    __layers : list of Layers
        stores the layers data
        (on default there is only the input_layer)
    """

    def __init__(self, input_size, cost_function=Loss):
        self.__layers = [InputLayer(input_size)]
        self.__cost_function = cost_function

    def get_layers(self):
        return self.__layers

    def get_cost_function(self):
        return self.__cost_function

    def add_layer(self, layer_size, activation_function=Identity):
        """
        Creates a new layer with its respective biases and full connections with the precedent layer
        """
        self.__layers.append(Layer(len(self.__layers[-1]), layer_size, activation_function))

    def calculate_cost_after_one_propagation(self, expected_output):
        return self.__cost_function.f(self.__layers[-1].get_layer_values(), expected_output)

    def forward_propagation(self, input_data):
        """
        Takes an input date and calculates the values of each neuron for each layer
        - we store that in a list and return it
        - we need those values in order to implement backpropagation

        Params
        ______

        input_data : list of floats
            a list of floats, containing the value for each neuron in the input layer

        Returns
        _______

        A list of layers, each layer containing the values of each neuron before activation
        - in __list we keep the values of each neuron after activation

        Raises
        ______

        NNError : if input_data's size doesn't correspond with the input layer's size

        """
        self.__layers[0].set_layer_values(input_data)
        result = []
        result.append(input_data)
        for i in range(1, len(self.__layers)):
            first_result = self.__layers[i].calculate_layer(self.__layers[i - 1].get_layer_values())
            self.__layers[i].set_layer_values(self.__layers[i].calculate_activated(first_result))
            result.append(first_result)
        return result

    def back_propagation(self, training_data):
        """
        Takes some training examples and updates the weights and biases in order to improve
        the performance of the neural net over that training data

        - in order to do that we need to find how much to increase or decrease each weight and bias
        - as the entire neural net is a function that takes some input data and returns some output
        data, with a lot of parameters as weights and biases

        Params
        ______

        training_data : a list of TrainingData objects
            Each training data contains an input and an expected output
        """
        simple_results = self.forward_propagation(training_data.input_data)
        delta = [None] * (len(self.__layers) - 1)
        biases_modifier = [None] * (len(self.__layers) - 1)
        weights_modifier = [None] * (len(self.__layers) - 1)
        delta[len(self.__layers) - 2] = self.__cost_function.df(
            self.__layers[len(self.__layers) - 1].get_layer_values(),
            training_data.output_data)
        for i in range(len(self.__layers) - 2, 0, -1):
            delta[i - 1] = np.matmul(self.__layers[i + 1].get_weights(),
                                     self.__layers[i + 1].calculate_derived(simple_results[i + 1]) * delta[i])
        for i in range(len(self.__layers) - 2, -1, -1):
            biases_modifier[i] = self.__layers[i + 1].calculate_derived(simple_results[i + 1]) * delta[i]
            weights_modifier[i] = np.outer(self.__layers[i].get_layer_values(),
                                           biases_modifier[i])
        return delta, weights_modifier, biases_modifier

    def train(self, training_data, epochs=1, batch_size=32,
              learning_rate=0.05, interpret_result=None):
        """
        Trains the neural net
        In other words modify the weights and biases in order to minimize the cost function
        Also shows in real time the cost and accuracy for each batch in each epoch
        After each epoch it shows in the end the average cost and accuracy for the entire training data

        Params
        ______

        training_data: a list of TrainingData
            A list used to store input data and expected output for each training set
        epochs: integer, optional (default set to 1)
            The number of the times we train over the training data
        batch_size: integer, optional (default set to 32)
            We split training data into batches, in order to speed up the training process
        learning_rate: float, preferably between (0,1] (default set to 0.05)
        interpret_result: A function
            It interprets the result from the output layer
            We use this function in order to calculate the accuracy

        """
        random.shuffle(training_data)
        training_batches = [training_data[i * batch_size:min((i + 1) * batch_size, len(training_data))]
                            for i in range(math.ceil(len(training_data) / batch_size))]
        for a in range(epochs):
            print(f"Epoch: {a + 1}")
            print("__________________________________")
            total_cost = 0
            total_accuracy = 0
            for batch in training_batches:
                cost = 0
                accuracy = 0
                batch_weights_modifier = []
                batch_biases_modifier = []
                for layer in self.__layers:
                    if isinstance(layer, Layer):
                        shape = layer.get_weights().shape
                        batch_weights_modifier.append(np.zeros(shape=shape))
                        batch_biases_modifier.append(np.zeros(len(layer)))
                for training_set in batch:
                    _, weights_modifier, biases_modifier = self.back_propagation(training_set)
                    cost += self.__cost_function.f(self.__layers[-1].get_layer_values(), training_set.output_data)[0]
                    if interpret_result is not None:
                        accuracy += 1 if interpret_result(self.__layers[-1].get_layer_values()) == interpret_result(training_set.output_data) else 0
                    batch_weights_modifier = [batch_weights_modifier[i] + weights_modifier[i]
                                              for i in range(len(self.__layers) - 1)]
                    batch_biases_modifier = [batch_biases_modifier[i] + biases_modifier[i]
                                             for i in range(len(self.__layers) - 1)]
                batch_biases_modifier = [batch_biases_modifier[i] / len(batch) for i in range(len(self.__layers) - 1)]
                batch_weights_modifier = [batch_weights_modifier[i] / len(batch) for i in range(len(self.__layers) - 1)]
                for i in range(len(self.__layers) - 1):
                    self.__layers[i + 1].update_weights(batch_weights_modifier[i] * learning_rate)
                    self.__layers[i + 1].update_biases(batch_biases_modifier[i] * learning_rate)
                    print(f"Cost: " + "{:.7f}".format(cost/len(batch)) +
                          f" --- Accuracy: {100*accuracy/len(batch)}%", end="\r")
                total_cost += cost
                total_accuracy += accuracy
            print(f"Cost: " + "{:.7f}".format(total_cost / len(training_data)) +
                  f" --- Accuracy: {100*total_accuracy/len(training_data)}%")

