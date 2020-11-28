import numpy as np


def shuffle_in_unison_scary(a, b):
    """
    A function that shuffles two arrays in unison
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

"""
def visualise(input, output, model):
    img,ax = plt.subplots(1,1)
    message = "I thought it looked like {} but it is actually {}"
    im = ax.imshow(input[0])
    for i in range(0, len(input)):
        im.set_data(input[i])
        img.canvas.draw_idle()
        predicted, prob = model.prediction(input[i])
        if predicted == output[i]:
            plt.pause(0.05)
        else:
            print(message.format(predicted,output[i]), end = "\r")
            plt.pause(3)
def see_diferences_over_alpha(start_value,end_value,step,epochs,layers,
                              n_input,n_momentum,input,output,t_in,t_out):
    np.random.seed(0)
    Y = np.arange(int((end_value-start_value)/step))
    X = np.empty(int((end_value-start_value)/step))
    Z = np.empty(int((end_value-start_value)/step))
    nr = 0
    for k in np.arange(start_value,end_value,step):
        model = NN.NN(n_input,k,n_momentum)
        for i in range(len(layers)):
            model.add_layer(layers[i][0],layers[i][1])
        model.add_layer(10,sigmoid)
        model.fit_model_on(input,output,epochs,32)
        X[nr] = model.accuracy(input,output)
        Z[nr] = model.accuracy(t_in,t_out)
        nr+=1
    plt.plot(Y,X)
    plt.plot(Y,Z)
    plt.show()
    """
"""
def load_model(name):
    with open(name,"r") as f:
        data = f.read().split()
        n_input = int(data[0])
        n_layers = int(data[1])
        n_learning_rate = float(data[2])
        n_momentum = float(data[3])
        model = NN.NN(n_input,n_learning_rate,n_momentum)
        pozition = 4
        while pozition < len(data):
            length = int(data[pozition])
            pozition+=1
            activation = data[pozition]
            pozition+=1
            model.add_layer(length,activation)
            #load weights
            for x in range(len(model.B[model.n_layers-1])):
                for y in range (len(model.B[model.n_layers])):
                    model.W[model.n_layers][y][x] = np.float64(data[pozition])
                    pozition+=1
            #load biases
            for x in range(len(model.B[model.n_layers])):
                model.B[model.n_layers][x] = np.float64(data[pozition])
                pozition+=1
    return model
    """
