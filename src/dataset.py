from typing import Any, List, Union, Tuple
from utils.data import WhiteBoxDataloader
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import torch
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchtext import datasets, data
import conf
from utils import utils
import pandas as pd
import random
import os
from utils.plotter import *

torch.manual_seed(17)


class CifarDataset:
    def __init__(self):
        pass

    def get_dataset(self, c):
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
        if c.n_classes == 100:
            trainset1 = CIFAR100(root='./data', train=True,
                                download=True, transform=transform1)
            # trainset2 = CIFAR100(root='./data', train=True,
            #                      download=True, transform=transform2)

            testset = CIFAR100(root='./data', train=False,        
                            download=True, transform=transform1)
        else:
            trainset1 = CIFAR10(root='./data', train=True,
                                download=True, transform=transform1)         

            testset = CIFAR10(root='./data', train=False,        
                            download=True, transform=transform1)

        trainset = trainset1  # + trainset2
        dataset = testset + trainset
        target_train, target_val, shadow_train, shadow_val = torch.utils.data.random_split(
            dataset,
            [c.target_train, c.target_val,
             c.shadow_train, c.shadow_val]
        )

        target_trainloader = DataLoader(
            target_train, batch_size=batch_size, shuffle=True, num_workers=2)
        target_valloader = DataLoader(
            target_val, batch_size=batch_size, shuffle=False, num_workers=2)
        shadow_trainloader = DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_valloader = DataLoader(
            shadow_val, batch_size=batch_size, shuffle=False, num_workers=2)

        dataloader = WhiteBoxDataloader(
            target_trainloader=target_trainloader,
            target_valloader=target_valloader,
            shadow_trainloader=shadow_trainloader,
            shadow_valloader=shadow_valloader
        )

        return dataloader


def get_CifarDataloader(c):
    cifar = CifarDataset()
    return cifar.get_dataset(c)


class UTKFaceDataset(Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "race", transform=None, target_transform=None) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.processed_path = os.path.join(self.root, 'processed.txt')
       
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []       
            
        with open(self.processed_path, 'r') as f:
            assert f is not None
            for i in f:
                image_name = i.split('jpg ')[0]
                attrs = image_name.split('_')
                if len(attrs) < 4 or int(attrs[2]) >= 4 or '' in attrs:
                    continue
                self.lines.append(image_name+'jpg')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(
            self.root, 'raw/', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)

            else:
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

def get_UTKDataloader():
    transform_one = transforms.Compose([            
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transform_two = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    trainset_one = UTKFaceDataset(root='./data/UTKFace/train', transform=transform_one)  
    trainset_two = UTKFaceDataset(root='./data/UTKFace/train', transform=transform_two)  
    testset = UTKFaceDataset(root='./data/UTKFace/test', transform=transform_one)  
    
    dataset = trainset_one + trainset_two + testset
    len_ = len(dataset) // 2 + 1    
    target_train, target_val, shadow_train, shadow_val = torch.utils.data.random_split(
            dataset,
            [int(len_*0.8)+1, int(len_*0.2),
            int(len_*0.8), int(len_*0.2)]            
        )
    batch_size = conf.batch_size
    target_trainloader = DataLoader(
        target_train, batch_size=batch_size, shuffle=True, num_workers=2)
    target_valloader = DataLoader(
        target_val, batch_size=batch_size, shuffle=False, num_workers=2)
    shadow_trainloader = DataLoader(
        shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
    shadow_valloader = DataLoader(
        shadow_val, batch_size=batch_size, shuffle=False, num_workers=2)

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
        self.train_logits_ = train_logits[len(
            test_logits)*2:len(test_logits)*4]
        self.train_logits = train_logits[:len(self.test_logits)]
        logits = self.init_label()
        self.logits, self.logit_labels = utils.split_data(logits)
        self.is_train = train

    def __len__(self):
        return len(self.logits)

    def init_label(self):        
        attack_data = []
        # dist_neg = torch.cdist(self.train_logits, self.test_logits)
        # dist_pos = torch.cdist(self.train_logits, self.train_logits_)

        for member, non_member in zip(self.train_logits, self.test_logits):
            attack_data.append((member.cpu().detach().numpy(), 1))
            attack_data.append((non_member.cpu().detach().numpy(), 0))

        return attack_data

    def __getitem__(self, idx):
        return (self.logits[idx], self.logit_labels[idx])


def prepare_attack_data(train_logits, test_logits):
    """Preparing data for non-NN attack model"""

    attack_data = []
    test_len = len(test_logits)
    train_logits_ = train_logits[:test_len]
    train_logits = train_logits[test_len:test_len*2]

    dist_neg = torch.cdist(train_logits, test_logits)
    dist_pos = torch.cdist(train_logits, train_logits_)

    # Labeling positive and negative samples
    for member, non_member in zip(dist_pos, dist_neg):
        attack_data.append((member.cpu().detach().numpy(), 1))
        attack_data.append((non_member.cpu().detach().numpy(), 0))

    random.shuffle(attack_data)
    plot_pred_diff(train_logits[:100], test_logits[:100])
    return attack_data



"""
class ImdbReviewsDataset:
    def __init__(self) -> None:
        self.vocab_size, self.n_clases, self.trainset, self.testset = self.init_voca()

    def init_voca(self):
        TEXT = data.Field(sequential=True, batch_first=True, lower=True)
        LABEL = data.Field(sequential=False, batch_first=True)
        trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(trainset, min_freq=5)
        LABEL.build_vocab(trainset)
        vocab_size = len(TEXT.vocab)
        n_classes = len(LABEL.vocab) - 1
        return vocab_size, n_classes, trainset, testset

    def get_dataset(self):
        target_train, target_test = self.trainset.split(split_ratio=0.8)
        shadow_train, shadow_test = self.testset.split(split_ratio=0.8)
        target_trainloader, shadow_trainloader, target_valloader, shadow_valloader = data.BucketIterator.splits(
            (target_train, shadow_train, target_test,
             shadow_test), batch_size=conf.batch_size,
            shuffle=True, repeat=False)

        dataloader = WhiteBoxDataloader(
            target_trainloader=target_trainloader,
            target_valloader=target_valloader,
            shadow_trainloader=shadow_trainloader,
            shadow_valloader=shadow_valloader
        )        
        return dataloader
"""    

