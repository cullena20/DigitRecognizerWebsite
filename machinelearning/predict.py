import keras
import os
import pickle
# this is needed when predict is imported, but breaks when this file is run directly
from machinelearning import mynet


def setup_my_nn():
    with open(os.path.join(os.getcwd(), 'machinelearning/models/my_nn/weights.pkl'), 'rb') as f:
        weights = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'machinelearning/models/my_nn/biases.pkl'), 'rb') as f:
        biases = pickle.load(f)
    sizes = [784, 30, 10]
    my_nn = mynet.NeuralNetwork(sizes, weights, biases)
    return my_nn

tf_nn = keras.models.load_model(os.path.join(os.getcwd(), "machinelearning/models/simple_keras_nn"))
cnn = keras.models.load_model(os.path.join(os.getcwd(), "machinelearning/models/cnn"))
my_nn = setup_my_nn()

def tfnn_prediction(image):
    pred = tf_nn.predict(image)
    return pred

def cnn_prediction(image):
    pred = cnn.predict(image)
    return pred

def mynn_prediction(image):
    image = image.reshape(784, 1)
    pred = my_nn.feedforward(image)
    return pred