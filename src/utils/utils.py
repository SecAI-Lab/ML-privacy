import os
import numpy as np

def validate_path(path):
    if os.path.exists(path):
        return True
    if not os.path.exists(path) and os.path(path).isdir():
        os.makedirs(path)
    elif not os.path.exists(path):
        raise ValueError("Path doesn't exist")

def get_labels(data):    
    labels = [label for _, label in data]    
    return np.concatenate(labels)

def split_data(data):
        x = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        return x, y
    
        