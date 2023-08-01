import torch
from dataclasses import dataclass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 20
batch_size = 32


@dataclass
class CifarConf:
    n_classes = 10
    target_train = 25000  # 60000 --> when augmented
    shadow_train = 25000  # 40000 --> when augmented
    target_val = 5000
    shadow_val = 5000
    data_type = f'cifar{n_classes}'
    target_path = f'../weights/target_{data_type}.pt'
    shadow_path = f'../weights/shadow_{data_type}.pt'
    nn_attacker_path = f'../weights/rnnAttack_{data_type}.pt'
    attacker_path = f'../weights/RF_attack_{data_type}.joblib'


@dataclass
class UTKFaceConf:
    n_classes = 4  # 4 -> race, 117 -> age, 3 -> gender
    data_type = f'utkface{n_classes}'
    target_path = f'./weights/target_{data_type}.pt'
    shadow_path = f'./weights/shadow_{data_type}.pt'
    nn_attacker_path = f'./weights/rnnAttack_{data_type}.pt'
    attackers_path = f'./weights/RF_attack_{data_type}.joblib'
