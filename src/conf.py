import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cifar_classes = 100
epochs = 10
batch_size = 32

# data sizes for splitting
unseen_test = 1000 # not used yet
target_train = 25000 #60000
shadow_train = 25000 #40000
target_val = 5000
shadow_val = 5000
