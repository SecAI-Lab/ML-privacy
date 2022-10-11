import os
import numpy as np

def validate_path(path):
    if not os.path.exists(path) and os.path(path).isdir():
        os.makedirs(path)
    elif not os.path.exists(path):
        raise ValueError("Path doesn't exist")

def get_labels(data):    
    labels = [label for _, label in data]    
    return np.concatenate(labels)
    
        