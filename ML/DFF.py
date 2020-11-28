import time as tm
from ML import utils
from ML.layers import *
import copy


class DFF(object):
    """
    A class used to represent a neural_network
    More exactly a sequential model

    Attributes
    __________

    input_size : integer, the number of neurons on the input layer
    layers : the list of layers
    n_layers : integer, the number of layers
    n_out : integer, the number of neurons on the output layer

    Methods
    _______

    add_layer : adds a given layer to the neural network
                - it needs to be a specific Layer from the layer module
    forward : does forward propagation for a given input and batch size
    backward : does backward propagation in order to optimize
     each value on each layer, to obtain a smaller loss
    compile : compiles the model, it need to be called after adding all layers
              it takes as parameters an optimizer from the optimizers module
              and a loss function from the losses module
    fit_model_on : trains the model for a given training data, batch size and validation
    evaluate : evaluates the average loss and accuracy on a test data
    """
    def __init__(self, input_size):
        self.__input_size = input_size
        self.__layers = []
        self.__n_layers = 1
        self.__n_out = input_size
        self.__layers.append(Flatten(input_size))

    def add_layer(self, layer):
        """
        Takes a layer and appends it to the layer list
        :param layer: a given layer
        """
        self.__layers.append(layer)
        self.__n_layers += 1
        self.__n_out = len(layer)

    def __forward(self, input, batch_size):
        """
        Does the forward propagation on the DFF
        :param input: The input elements of size(batch_size, input_size)
        :param batch_size: The size of the batch
        :return: The result of the forward propagation, being the values on the last layer
        """
        result = input
        for i in range(self.__n_layers):
            result = self.__layers[i](result, batch_size=batch_size)
        return result

    def __backprop(self, input, output, batch_size):
        """
        Takes each layer in reverse order, after computing the forward propagation
        and computes its gradient in order to minimize the loss function after subtracting
        the gradient from the actual weights
        :param input: the input batch
        :param output: the output batch
        :param batch_size: the batch size
        """
        result = self.__forward(input, batch_size)
        self.acc += np.sum(np.argmax(result, axis=1) == output)
        expected_output = np.zeros(shape=(batch_size, self.__n_out)).astype(np.float32)
        expected_output[np.arange(batch_size), output] = 1.0
        cost, delta = self.__loss(result, expected_output)
        self.cost += cost
        for i in range(self.__n_layers-1, -1, -1):
            if self.__layers[i].trainable:
                delta = self.__layers[i].train(inputs=self.__layers[i-1].activated, delta=delta)

    def compile(self, optimizer, loss):
        """
        Compiles the neural net, adding to each layer the optimizer and setting the loss function
        as the given one
        :param optimizer: an optimizer from the optimizers module
        :param loss: a loss function from the losses module
        """
        self.__loss = loss
        for i in range(self.__n_layers):
            if self.__layers[i].trainable:
                self.__layers[i].build(copy.deepcopy(optimizer), len(self.__layers[i-1]))

    def fit_model_on(self, input, output, epochs=5, batch_size=32, validation_scale=0, val_in=None, val_out=None):
        """

        This is pretty much the way in which the model interacts with the user,
        as it shows in real time how the neural network is performing

        :param input: the training data input
        :param output: the training data output
            (must be a single integer representing which neuron should be the most active after forward prop)
        :param epochs: integer, the number of times we are going to train it over the training data
                (default is 5)
        :param batch_size: integer, the size of the batch (default is 32)
        :param validation_scale: float between 0 and 1, it is used to split the training data
                into training and validation, optional
        :param val_in: validation input, optional
        :param val_out: validation output, optional
        """

        train_acc = np.empty(epochs)
        val_acc = np.empty(epochs)
        train_loss = np.empty(epochs)
        val_loss = np.empty(epochs)
        if val_in is None and validation_scale > 0:
            utils.shuffle_in_unison_scary(input, output)
            val_in = input[:int(len(input) * validation_scale)]
            val_out = output[:int(len(output) * validation_scale)]
            input = input[int(len(input) * validation_scale):]
            output = output[int(len(output) * validation_scale):]
        message = "Test {}/{} --- accuracy:{}% --- Loss:{}"
        end_message = "Test {}/{} --- accuracy:{}% --- Loss:{} --- Time:{}"
        end_message_validation = "Test {}/{} --- accuracy:{}% --- Loss:{} --- Val_acc:{}% --- Val_Loss:{} --- Time:{}"
        utils.shuffle_in_unison_scary(input, output)
        for k in range(epochs):
            print("Epoch {}/{}".format(k + 1, epochs))
            time = tm.time()
            epoch_acc = 0
            epoch_cost = 0
            for i in range(0, len(input) // batch_size):
                self.cost = 0
                self.acc = 0
                length = min(i + batch_size, len(input)) - i
                self.__backprop(input[i * batch_size:(i + 1) * batch_size],
                              output[i * batch_size:(i + 1) * batch_size], batch_size)
                print(message.format("{:5.0f}".format(i), len(input) // batch_size,
                                     "{:7.2f}".format(100 * self.acc / length),
                                     "{:7.4f}".format(self.cost / (2 * length))), end="\r")
                epoch_acc += self.acc
                epoch_cost += self.cost
            elapsed_time = tm.time() - time

            train_acc[k] = 100 * epoch_acc / len(input)
            train_loss[k] = epoch_cost / len(input)
            if val_in is not None:
                val_acc[k], val_loss[k] = self.evaluate(val_in, val_out)
                print(end_message_validation.format(len(input) // batch_size, len(input) // batch_size,
                                                    "{:7.2f}".format(100 * epoch_acc / len(input)),
                                                    "{:7.4f}".format(epoch_cost / len(input)),
                                                    "{:7.2f}".format(val_acc[k]),
                                                    "{:7.4f}".format(val_loss[k]),
                                                    tm.strftime("%H:%M:%S", tm.gmtime(elapsed_time))))
            else:
                print(end_message.format(len(input) // batch_size, len(input) // batch_size,
                                         "{:7.2f}".format(100 * epoch_acc / len(input)),
                                         "{:7.4f}".format(epoch_cost / len(input)),
                                         tm.strftime("%H:%M:%S", tm.gmtime(elapsed_time))))
        if val_in is None:
            return train_acc, train_loss
        else:
            return train_acc, train_loss, val_acc, val_loss

    def evaluate(self, input, output):
        """
        Given an input and an expected output this function calculates the average loss and accuracy
        :param input: the input data
        :param output: the output data
        :return: a tuple of accuracy and loss
        """
        for layer in self.__layers:
            if isinstance(layer, Dropout):
                layer.set_dropout(False)
        result = self.__forward(input, len(input))
        for layer in self.__layers:
            if isinstance(layer, Dropout):
                layer.set_dropout(True)
        accuracy = np.sum(np.argmax(result, axis=1) == output)
        expected_output = np.zeros(shape=(len(input), self.__n_out)).astype(np.float32)
        expected_output[np.arange(len(input)), output] = 1.0
        loss = self.__loss(result, expected_output)[0]
        return 100 * accuracy / len(input), loss / len(input)
