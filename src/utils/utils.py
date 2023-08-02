import os
import numpy as np


def validate_path(path):
    if os.path.exists(path):
        return True
    else:
        os.makedirs('../weights')


def get_labels(data):
    labels = [label for _, label in data]
    return np.concatenate(labels)


def split_data(data):
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    return x, y
