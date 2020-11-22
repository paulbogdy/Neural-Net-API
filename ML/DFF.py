import numpy as np
import time as tm
import random as rd
import matplotlib.pyplot as plt
from ML import utils
from ML.optimizers import *
from ML.activations import *
import multiprocessing
import os


class DFF(object):
    """description of class"""

    def __init__(self, input_size):
        self.activated_neurons = [np.empty(input_size)]
        self.neurons = [np.empty(input_size)]
        self.val_loss = 0
        self.input_size = input_size
        self.weights = []
        self.biases = []
        self.f = []
        self.b = []
        self.w = []
        self.n_layers = 1
        self.cost = 0
        self.acc = 0
        self.n_out = input_size
        self.delta = []

    def add_layer(self, length, activation):
        self.n_layers += 1
        self.activated_neurons.append(np.empty(length).astype(np.float32))
        self.neurons.append(np.empty(length).astype(np.float32))
        self.delta.append(np.empty(length).astype(np.float32))
        self.weights.append(0.22 * (np.random.randn(self.n_out, length) - 0.05).astype(np.float32))
        self.biases.append(np.zeros(length).astype(np.float32))
        self.w.append(np.zeros((self.n_out, length)).astype(np.float32))
        self.b.append(np.zeros(length).astype(np.float32))
        self.f.append(activation)
        self.n_out = length

    def forward(self, input, batch_size):
        self.activated_neurons[0] = np.array(input.reshape(batch_size, self.input_size)).astype(np.float32)
        for i in range(1, self.n_layers):
            np.dot(self.activated_neurons[i - 1], self.weights[i - 1], self.neurons[i])
            self.neurons[i] += self.biases[i - 1]
            self.activated_neurons[i], self.neurons[i] = self.f[i - 1](self.neurons[i])

    def backprop(self, output, batch_size):
        self.acc += np.sum(np.argmax(self.activated_neurons[self.n_layers - 1], axis=1) == output)
        expected_output = np.zeros(shape=(batch_size, self.n_out)).astype(np.float32)
        expected_output[np.arange(batch_size), output] = 1.0
        cost, self.delta[self.n_layers - 2] = self.loss(self.activated_neurons[self.n_layers - 1]
                                                                , expected_output)
        self.cost += cost
        """
        for i in range(self.n_layers - 2, 0, -1):
            self.delta[i] = np.einsum('ij,ij->ij', self.neurons[i + 1], self.delta[i])
            self.delta[i].dot(self.weights[i].T, self.delta[i - 1])
        self.delta[0] = np.einsum('ij,ij->ij', self.delta[0], self.neurons[1])
        """
        for j in range(batch_size):
            for i in range(self.n_layers - 2, 0, -1):
                self.delta[i][j] *= self.neurons[i + 1][j]
                self.delta[i][j].dot(self.weights[i].T, self.delta[i - 1][j])
            self.delta[0][j] *= self.neurons[1][j]

        for i in range(self.n_layers - 2, -1, -1):
            self.b[i] = np.einsum('ij->j', self.delta[i])
            self.w[i] = np.einsum('ij,ik->jk', self.activated_neurons[i], self.delta[i])

    def compile(self, optimizer, loss):
        optimizer.fit(self.weights)
        self.optimizer = optimizer
        self.loss = loss

    def fit_model_on(self, input, output, epochs=5, batch_size=32, validation_scale=0, val_in=None, val_out=None):
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
        for k in range(epochs):
            self.neurons[0] = np.empty(shape=(batch_size, self.input_size))
            self.activated_neurons[0] = np.empty(shape=(batch_size, self.input_size))
            for i in range(1, self.n_layers):
                self.neurons[i] = np.empty(shape=(batch_size, len(self.biases[i - 1]))).astype(np.float32)
                self.activated_neurons[i] = np.empty(shape=(batch_size, len(self.biases[i - 1]))).astype(np.float32)
            for i in range(self.n_layers - 1):
                self.delta[i] = np.empty(shape=(batch_size, len(self.biases[i]))).astype(np.float32)
            utils.shuffle_in_unison_scary(input, output)
            print("Epoch {}/{}".format(k + 1, epochs))
            time = tm.time()
            epoch_acc = 0
            epoch_cost = 0
            for i in range(0, len(input) // batch_size):
                self.cost = 0
                self.acc = 0
                length = min(i + batch_size, len(input)) - i
                self.forward(input[i * batch_size:(i + 1) * batch_size], batch_size)
                self.backprop(output[i * batch_size:(i + 1) * batch_size], batch_size)
                for j in range(self.n_layers - 2, -1, -1):
                    self.optimizer.optimize(weights=self.weights[j], biases=self.biases[j],
                                            weights_modifier=self.w[j]/batch_size, biases_modifier=self.b[j]/batch_size,
                                            index=j)
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
    """
    def save_model(self, name):
        with open(name, "w") as f:
            # date simple
            f.write('%s' % self.input_size + ' ')
            f.write('%s' % self.n_layers + ' ')
            f.write('%s' % self.learning_rate + ' ')
            f.write('%s' % self.n_momentum + ' ')
            # afisez layerele
            for i in range(1, self.n_layers - 1 + 1):
                # lungimea
                f.write('%s' % len(self.biases[i]) + ' ')
                # activare
                f.write(self.f[i] + ' ')
                # afisez weight-urile
                for x in range(len(self.biases[i - 1])):
                    for y in range(len(self.biases[i])):
                        f.write('%s' % self.weights[i][y][x] + ' ')
                # afisez bias-urile
                for x in range(len(self.biases[i])):
                    f.write('%s' % self.biases[i][x] + ' ')
    """
    def evaluate(self, input, output):
        self.neurons[0] = np.empty(shape=(len(input), self.input_size))
        self.activated_neurons[0] = np.empty(shape=(len(input), self.input_size))
        for i in range(1, self.n_layers):
            self.neurons[i] = np.empty(shape=(len(input), len(self.biases[i - 1]))).astype(np.float32)
            self.activated_neurons[i] = np.empty(shape=(len(input), len(self.biases[i - 1]))).astype(np.float32)
        for i in range(self.n_layers - 1):
            self.delta[i] = np.empty(shape=(len(input), len(self.biases[i]))).astype(np.float32)
        self.forward(input, len(input))
        accuracy = np.sum(np.argmax(self.activated_neurons[self.n_layers - 1], axis=1) == output)
        expected_output = np.zeros(shape=(len(input), self.n_out)).astype(np.float32)
        expected_output[np.arange(len(input)), output] = 1.0
        loss = self.loss(self.activated_neurons[self.n_layers - 1], expected_output)[0]
        return 100 * accuracy / len(input), loss / len(input)
