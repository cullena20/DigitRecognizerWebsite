import keras
import os
import numpy as np

def tfnn_prediction(image):
    model = keras.models.load_model(os.path.join(os.getcwd(), "machinelearning/models/simple_keras_nn"))
    pred = model.predict(image)
    return np.argmax(pred)

def cnn_prediction(image):
    model = keras.models.load_model(os.path.join(os.getcwd(), "machinelearning/models/cnn"))
    pred = model.predict(image)
    return np.argmax(pred)

def make_prediction(image):
    model = keras.models.load_model(os.path.join(os.getcwd(), "machinelearning/models/simple_keras_nn"))
    pred = model.predict(image)
    return np.argmax(pred)