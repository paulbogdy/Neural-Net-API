from ML.nets import *


def interpretation(arr):
    return 1 if 1-arr < arr else 0


if __name__ == '__main__':
    """
    Here is an example on how the neural net learns to solve the xor function
    """
    nn = DeepFeedForward(2)
    nn.add_layer(2, Sigmoid)
    nn.add_layer(1, Tanh)
    training_data = [TrainingData([0, 0], 0),
                     TrainingData([0, 1], 1),
                     TrainingData([1, 0], 1),
                     TrainingData([1, 1], 0)]
    nn.train(training_data, epochs=40000, batch_size=4, learning_rate=0.5, interpret_result=interpretation)
