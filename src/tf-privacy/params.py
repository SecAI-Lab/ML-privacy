from dataclasses import dataclass
import numpy as np


@dataclass
class PredictionData:
    logits_train: np.array
    logits_test: np.array
    loss_train: np.array
    loss_test: np.array
    train_labels: np.array
    test_labels: np.array


class Configs:
    threshold = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    batch_size = 250
    epochs = 10
    input_shape = (224, 224, 3)
    num_classes = 100
