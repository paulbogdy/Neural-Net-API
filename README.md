# Minimalist ML library
This is mostly a project intended for learning the basics of neural networks. I implemented a generalised Deep Feed Forward with an easy to use api, kind of similar with what Tensorflow Keras has.
In main there is an example use of this library to create a simple DFF in order to solve the MNIST handwritten dataset problem.

If you want to learn how Neural Networks work, this git is a perfect example. There are not so many lines of code, and a lot of comments. Check it out.

    n_input = 28 * 28
    n_learning_rate = 0.01
    momentum = 0.5
    epochs = 20

    [train_image, train_result, test_image, test_result] = mnist.read_mnist()

    model = DFF(n_input)

    model.add_layer(layer=ML.layers.Dense(256, activation=Relu()))
    model.add_layer(layer=ML.layers.Dense(256, activation=Relu()))
    model.add_layer(layer=ML.layers.Dense(10, activation=Sigmoid()))

    model.compile(optimizer=SGD(learning_rate=n_learning_rate, momentum=momentum), loss=MeanSquaredError())

    acc, loss, v_acc, v_loss = model.fit_model_on(train_image, train_result, epochs=epochs, batch_size=32,
                                                  val_in=test_image, val_out=test_result)

    print(model.evaluate(test_image, test_result)[0], "%")
    
For this library a manual differentiation was used. (It is impractical for bigger projects, but as this one is for learning purposes, manual differentiation might be the best way to learn about how Neural Networks work).

Features:

-Models: DFF (similar with a sequential model, I named it DFF because that's pretty much the entire usage of it, as it is now)

-Activation Functions (used for layers): Sigmoid, Relu, Tanh, Softmax

-Initializers (used to initialize weights and biases): Zeroes(full of 0), RandomNormal(normal distributaion), RandomUniform(uniform distribution) - for parameters check the code in initializers.py

-Layers: Flatten, Dense, Dropout

-Losses: MSE(Mean Squared Error), MAE(Mean Absolute Error)

-Optimizers: SGD(Can use Nestrov optimisation for momentum)

