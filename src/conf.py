import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cifar_classes = 10
epochs = 10
batch_size = 32

# data sizes for splitting
unseen_test = 1000  # not used yet
target_train = 25000  # 60000
shadow_train = 25000  # 40000
target_val = 5000
shadow_val = 5000
cifar_target_path = './weights/target_cifar10.pt'
cifar_shadow_path = './weights/shadow_cifar10.pt'
attackers_path = {
    'rf_path': './weights/RF_attack_cifar10.joblib',
    'lr_path': './weights/LR_attack_cifar10.joblib',
    'mlp_path': './weights/MLP_attack_cifar10.joblib',
    'knn_path': './weights/KNN_attack_cifar10.joblib',
}
# rnn configs
n_layers = 1
hidden_dim = 256
emdeb_dim = 128
imdb_target_path = './weights/target_imdb.pt'
imdb_shadow_path = './weights/shadow_imdb.pt'

attack_for_rnn = './weights/RF_attack_rnn.joblib'
