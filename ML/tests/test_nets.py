import unittest
from ML.nets import *


class TestDeepFeedForward(unittest.TestCase):
    def test_init(self):
        net = DeepFeedForward(input_size=2)
        self.assertEqual(len(net.get_layers()), 1)
        self.assertEqual(len(net.get_layers()[0]), 2)

    def test_add_layer(self):
        net = DeepFeedForward(input_size=2)
        net.add_layer(16)
        self.assertEqual(len(net.get_layers()), 2)
        self.assertEqual(len(net.get_layers()[1]), 16)
        net.add_layer(2, Sigmoid)
        self.assertEqual(net.get_layers()[2].get_activation_function(), Sigmoid)

    def test_forward_propagation(self):
        net = DeepFeedForward(input_size=2)
        net.add_layer(2, Sigmoid)
        net.add_layer(1)
        input_val = np.array([5, 6])
        layers = net.get_layers()
        result_layers = net.forward_propagation(input_val)
        rez1 = 4.14410994
        rez2 = 9.03538307
        rez3 = 1.4508485866334874011841398927802
        self.assertTrue(abs(result_layers[1][0] - rez1) < 0.0001)
        self.assertTrue(abs(result_layers[1][1] - rez2) < 0.0001)
        self.assertTrue(abs(result_layers[2][0] - rez3) < 0.0001)
        self.assertTrue(abs(layers[1].get_layer_values()[0] - Sigmoid.f(rez1)) < 0.0001)
        self.assertTrue(abs(layers[1].get_layer_values()[1] - Sigmoid.f(rez2)) < 0.0001)
        self.assertTrue(abs(layers[2].get_layer_values()[0] - Identity.f(rez3)) < 0.0001)
        #print(Sigmoid.df(result_layers[1][0]), Sigmoid.df(result_layers[1][1]), Identity.df(result_layers[2][0]))
        #print(layers[1].get_layer_values()[0], layers[1].get_layer_values()[1], layers[2].get_layer_values()[0])

    def test_backward_propagation(self):
        net = DeepFeedForward(input_size=2)
        net.add_layer(2, Sigmoid)
        net.add_layer(1)
        #print(net.get_layers()[2].get_weights())
        # pt xor - lmao
        deltas = net.back_propagation(TrainingData([5, 6], 3))
        da20 = 2*(1.4508485866334873 - 3)
        dw200 = 0.9843899924706113*da20
        dw210 = 0.9998808946525293*da20
        db20 = da20
        da10 = 0.29624916*da20
        da11 = 0.80906772*da20
        dw100 = 5*0.01536633519432111*da10
        dw110 = 6*0.01536633519432111*da10
        dw101 = 5*0.00011909116138693098*da11
        dw111 = 6*0.00011909116138693098*da11
        db10 = 0.01536633519432111*da10
        db11 = 0.00011909116138693098*da11

        #print(dw100, dw101, dw110, dw111)
        #print(dw200, dw210)
        #print(deltas[0])
        #print(deltas[1])
        #print(deltas[2])

        self.assertTrue(abs(deltas[0][1][0] - da20) < 0.0001)
        self.assertTrue(abs(deltas[0][0][0] - da10) < 0.0001)
        self.assertTrue(abs(deltas[0][0][1] - da11) < 0.0001)

        self.assertTrue(abs(deltas[1][0][0][0] - dw100) < 0.0001)
        self.assertTrue(abs(deltas[1][0][0][1] - dw101) < 0.0001)
        self.assertTrue(abs(deltas[1][0][1][0] - dw110) < 0.0001)
        self.assertTrue(abs(deltas[1][0][1][1] - dw111) < 0.0001)
        self.assertTrue(abs(deltas[1][1][0][0] - dw200) < 0.0001)
        self.assertTrue(abs(deltas[1][1][1][0] - dw210) < 0.0001)

        self.assertTrue(abs(deltas[2][0][0] - db10) < 0.0001)
        self.assertTrue(abs(deltas[2][0][1] - db11) < 0.0001)
        self.assertTrue(abs(deltas[2][1][0] - db20) < 0.0001)


if __name__ == '__main__':
    unittest.main()
