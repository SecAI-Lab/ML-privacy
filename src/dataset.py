from utils.data import WhiteBoxDataloader
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchtext import datasets, data
import conf
from utils import utils
import pandas as pd
import random

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
        trainset1 = CIFAR10(root='./data', train=True,
                            download=True, transform=transform1)
        # trainset2 = CIFAR100(root='./data', train=True,
        #                      download=True, transform=transform2)

        testset = CIFAR10(root='./data', train=False,
                          download=True, transform=transform1)
        trainset = trainset1  # + trainset2
        dataset = testset + trainset
        target_train, target_val, shadow_train, shadow_val = torch.utils.data.random_split(
            dataset,
            [conf.target_train, conf.target_val,
                conf.shadow_train, conf.shadow_val]
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
        #print(len(target_train), len(target_test))
        return dataloader


class AttackDataset(Dataset):
    def __init__(self, train_logits, train_grads, test_logits, test_grads, train=True) -> None:
        super().__init__()
        self.test_logits = torch.cat([test_logits, test_logits])
        self.test_grads = test_grads[:len(self.test_logits)]
        self.train_grads = train_grads[:len(self.test_grads)]
        self.train_logits_ = train_logits[len(
            test_logits)*2:len(test_logits)*4]
        self.train_logits = train_logits[:len(test_logits)*2]
        logits, grads = self.init_label()
        self.logits, self.logit_labels = utils.split_data(logits)
        self.grads, self.grad_labels = utils.split_data(grads)
        self.is_train = train

    def __len__(self):
        return len(self.logit_labels)

    def init_label(self):
        # TODO: optimize by combining splitter
        logits, grads = [], []
        print(self.test_grads.shape, self.train_grads.shape)
        for member, non_member in zip(self.train_logits, self.test_logits):
            logits.append((member.cpu().detach().numpy(), 1))
            logits.append((non_member.cpu().detach().numpy(), 0))

        for member_, non_member_ in zip(self.train_grads, self.test_grads):
            grads.append((member_.cpu().detach().numpy(), 1))
            grads.append((non_member_.cpu().detach().numpy(), 0))

        return logits, grads

    def __getitem__(self, idx):
        return (
            (self.logits[idx], self.logit_labels[idx]),
            (self.grads[idx], self.grad_labels[idx])
        )


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
    # plot_pred_diff(dist_neg[:50], dist_pos[:50])
    return attack_data


