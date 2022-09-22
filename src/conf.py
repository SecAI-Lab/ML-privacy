import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cifar_classes = 10
epochs = 2