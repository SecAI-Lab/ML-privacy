import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random
from torchvision.datasets import CIFAR100, STL10
from torch.utils.data import DataLoader
from data import WhiteBoxDataloader
import conf
from utils.plotter import plot_pred_diff

torch.manual_seed(17)

class Cifar100Dataset:
    def __init__(self):
        pass


    def get_dataset(self):
        transform1 = transforms.Compose(
            [
                transforms.Resize((224, 224)),            
                transforms.ToTensor(),            
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        transform2 = transforms.Compose(
            [   
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),            
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        batch_size = conf.batch_size
        trainset1 = CIFAR100(root='./data', train=True,
                                                download=True, transform=transform1)    
        trainset2 = CIFAR100(root='./data', train=True,
                                                download=True, transform=transform2)                
                                    
        testset = CIFAR100(root='./data', train=False,
                                            download=True, transform=transform1)
        trainset = trainset1 #+ trainset2
        dataset = testset + trainset
        target_train, target_val, shadow_train, shadow_val = torch.utils.data.random_split(
                                                                        dataset, 
                                                                        [conf.target_train, conf.target_val, conf.shadow_train, conf.shadow_val]
                                                                       )
                                                                            
        target_trainloader = DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=2)
        target_valloader = DataLoader(target_val, batch_size=batch_size, shuffle=False, num_workers=2)
        shadow_trainloader = DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_valloader = DataLoader(shadow_val, batch_size=batch_size, shuffle=False, num_workers=2)        
        
        dataloader = WhiteBoxDataloader(
            target_trainloader=target_trainloader,
            target_valloader=target_valloader,
            shadow_trainloader=shadow_trainloader,
            shadow_valloader=shadow_valloader            
        )   

        return dataloader

class STL10Dataset:
    def __init__(self) -> None:
        pass

    def get_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = STL10(root='./data', split='train', transform=transform, download=True)
        test_set = STL10(root='./data', split='test', transform=transform, download=True)                
        batch_size=64
        dataset = train_set + test_set     
        
        length = len(dataset)
        each_length = length//4
        target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])

        train_loader = DataLoader(
            target_train, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(
            target_test, batch_size=64, shuffle=True, num_workers=2)

        shadow_trainloader = DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_testloader = DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)

        dataloader = WhiteBoxDataloader(
            target_trainloader=train_loader,
            target_valloader=test_loader,
            shadow_trainloader=shadow_trainloader,
            shadow_valloader=shadow_testloader            
        )  
        print(len(train_loader), len(test_loader), len(shadow_trainloader), len(shadow_testloader))
        return dataloader


def prepare_attack_data(train_logits, test_logits):          
    """Preparing data for non-NN attack model"""
    
    attack_data = []            
    train_logits_ = train_logits[:len(test_logits)]
    train_logits = train_logits[len(test_logits):len(test_logits)*2]  
   
    dist_neg = torch.cdist(train_logits, test_logits)
    dist_pos = torch.cdist(train_logits, train_logits_)
    
    # Labeling positive and negative samples
    for member, non_member in zip(dist_pos, dist_neg):                
        attack_data.append((member.cpu().detach().numpy(), 1))
        attack_data.append((non_member.cpu().detach().numpy(), 0))           
        
    random.shuffle(attack_data)
    # plot_pred_diff(dist_neg[:50], dist_pos[:50])           
    return attack_data 

def prepare_nn_attack_data(train_logits, test_logits):
    """Preparing data for NN attack model"""
    
    pos_sample = []
    neg_sample = []
    train_logits_ = train_logits[:len(test_logits)]
    train_logits = train_logits[len(test_logits):len(test_logits)*2]    
    
    # pos_sample = torch.stack([train_logits, train_logits_], dim=0)
    # neg_sample = torch.stack([train_logits, test_logits], dim=0)
    # print('pos samepl', pos_sample.shape)
    
    return train_logits, train_logits_, test_logits

