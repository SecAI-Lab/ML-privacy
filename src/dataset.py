import tensorflow as tf
import numpy as np


def load_cifar10():
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels
