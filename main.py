from ML.DFF import DFF
from ML.optimizers import *
from ML.losses import *
from ML.activations import *
from ML.datasets import mnist


if __name__ == '__main__':
    # variable
    n_input = 28 * 28
    n_learning_rate = 0.5
    momentum = 0.5
    epochs = 15

    [train_image, train_result, test_image, test_result] = mnist.read_mnist()

    model = DFF(n_input)

    model.add_layer(64, activation=Sigmoid())
    model.add_layer(64, activation=Sigmoid())
    model.add_layer(10, activation=Sigmoid())

    model.compile(optimizer=SGD(learning_rate=n_learning_rate, momentum=momentum), loss=MeanSquaredError())

    acc, loss, v_acc, v_loss = model.fit_model_on(train_image, train_result, epochs=epochs, batch_size=32,
                                                  val_in=test_image, val_out=test_result)

    print(model.evaluate(test_image, test_result)[0], "%")
