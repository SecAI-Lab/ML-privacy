import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.datasets import CIFAR100, CelebA
from torch.utils.data import DataLoader, Dataset
from data import WhiteBoxDataloader
import conf
from utils import utils

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


class AttackDataset(Dataset):
    def __init__(self, train_logits, test_logits, train=True) -> None:
        super().__init__()
        self.test_logits = torch.cat([test_logits, test_logits])
        self.train_logits_ = train_logits[len(test_logits)*2:len(test_logits)*4] 
        self.train_logits = train_logits[:len(test_logits)*2]
        data = self.init_label()
        self.data, self.labels = utils.split_data(data)              
        self.is_train = train        
    
    def __len__(self):       
        return len(self.labels)    
    
    def init_label(self):
        #TODO: optimize by combining splitter
        attack_data = []        
        for member, non_member in zip(self.train_logits, self.test_logits):                
            attack_data.append((member.cpu().detach().numpy(), 1))
            attack_data.append((non_member.cpu().detach().numpy(), 0)) 
        return attack_data

    def __getitem__(self, idx):                       
        return (self.data[idx], self.labels[idx])



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
        
    # random.shuffle(attack_data)
    # plot_pred_diff(dist_neg[:50], dist_pos[:50])           
    return attack_data 



"""
Not used

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

"""

