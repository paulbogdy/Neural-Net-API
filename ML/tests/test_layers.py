import random
import unittest
from ML.components.layers import *
from ML.handlers.errors import LayerError


class TestInputLayer(unittest.TestCase):
    def test_init(self):
        layer = InputLayer(3)
        self.assertEqual(len(layer), 3)
        layer.set_layer_values([1, 2, 3])
        self.assertEqual(layer.get_layer_values(), [1, 2, 3])
        try:
            layer.set_layer_values([1, 2, 3, 4])
            self.assertTrue(False)
        except LayerError:
            self.assertTrue(True)
        try:
            layer.set_layer_values([1, 2])
            self.assertTrue(False)
        except LayerError:
            self.assertTrue(True)


class TestLayer(unittest.TestCase):
    def test_init(self):
        layer = Layer(6, 9)
        self.assertEqual(len(layer), 9)
        self.assertEqual(len(layer.get_weights()), 6)
        self.assertEqual(len(layer.get_weights()[0]), 9)
        self.assertEqual(len(layer.get_biases()), 9)
        self.assertEqual(len(layer.get_layer_values()), 9)
        self.assertEqual(layer.get_activation_function(), Identity)
        layer = Layer(6, 9, Sigmoid)
        self.assertEqual(layer.get_activation_function(), Sigmoid)

    def test_calculate_layer(self):
        layer = Layer(2, 1)
        weights = layer.get_weights()
        bias = layer.get_biases()[0]
        result = layer.calculate_layer([5, 6])[0]
        expected_result = 6.68590462762553934
        self.assertTrue(abs(result - expected_result) < 0.00001)
        input_layer = InputLayer(3)
        try:
            layer.calculate_layer(input_layer.get_layer_values())
            self.assertTrue(False)
        except LayerError:
            self.assertTrue(True)

    def test_activated(self):
        layer = Layer(2, 2, Sigmoid)
        result = layer.calculate_layer(np.array([6, 9]))
        actual_result = layer.calculate_activated(result)
        self.assertTrue(abs(actual_result[0] - layer.get_activation_function().f(result[0])) < 0.0001)
        self.assertTrue(abs(actual_result[1] - layer.get_activation_function().f(result[1])) < 0.0001)

    def test_derived(self):
        layer = Layer(2, 2, Sigmoid)
        result = layer.calculate_layer(np.array([6, 9]))
        actual_result = layer.calculate_derived(result)
        self.assertTrue(abs(actual_result[0] - layer.get_activation_function().df(result[0])) < 0.0001)
        self.assertTrue(abs(actual_result[1] - layer.get_activation_function().df(result[1])) < 0.0001)


if __name__ == '__main__':
    unittest.main()
