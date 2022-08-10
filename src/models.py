import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model


def cnn_model():
    """ Define a Keras model without much of regularization
    Such a model is prone to overfitting"""
    shape = (32, 32, 3)
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), activation='relu')(i)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(10)(x)
    model = Model(i, x)
    return model
