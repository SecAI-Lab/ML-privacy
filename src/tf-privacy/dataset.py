import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (224, 224))
    return image, label


def load_cifar100():
    (train_data, train_labels), (test_data,
                                 test_labels) = tf.keras.datasets.cifar100.load_data()

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    train_data = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=100, drop_remainder=False))
    test_data = (test_ds
                 .map(process_images)
                 .shuffle(buffer_size=test_ds_size)
                 .batch(batch_size=10, drop_remainder=False))

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels


def load_texas100():
    data_files = '../dataset/dataset_texas/texas/100'        
    features = []
    labels = []
    for file in os.listdir(data_files):
        path = '{}/{}'.format(data_files, file)        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                splitted_line = line.split(",")
            if file == 'feats':
                features.append(list(map(int, splitted_line[1:])))
                data = np.array(features)
            else:   
                labels.append(int(splitted_line[0]) - 1)
                lbls = to_categorical(labels)

    print(data.shape, lbls)           
            

load_texas100()