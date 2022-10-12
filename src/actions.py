from unittest import TestLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import conf
from utils.plotter import *
from utils.utils import validate_path, split_data
from models.attack import ContrastiveLoss, SupervisedContrastiveLoss
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def train_model(dataloader, model, mode='target'):
    criterion = nn.CrossEntropyLoss()        
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainloader = dataloader.target_trainloader
    testloader = dataloader.target_valloader
    losses = {'train': [], 'val': []}
    path = f'./weights/{mode}_cifar100.pt'
    #validate_path('./weights')

    if mode == 'shadow':
        print("Training shadow model...")
        trainloader = dataloader.shadow_trainloader
        testloader = dataloader.shadow_valloader

    for epoch in tqdm(range(conf.epochs)):
        print('Epoch {}'.format(epoch+1))
        model.train()  
        train_loss = 0.0
        train_acc = 0           
    
        for i, (inputs, target) in enumerate(trainloader):
            inputs = inputs.to(conf.device)
            target = target.to(conf.device)                
            optimizer.zero_grad()
            pred = model(inputs)                        
            loss = criterion(pred, target) 
            _, preds = torch.max(pred, 1)
            
            loss.backward()
            optimizer.step()

            if i % 50 == 0:                
                val_loss, val_acc = test_model(testloader, model, mode='train')
            
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(preds == target.data)
            
        train_losses = train_loss / len(trainloader.dataset)
        train_accs = train_acc.double() / len(trainloader.dataset)

        val_losses = val_loss / len(testloader.dataset)
        val_accs = val_acc.double() / len(testloader.dataset)
        #losses['train'].append(train_losses)
        #losses['val'].append(val_losses)
        #plot_loss(losses, mode)
        print('Train Loss: {:.4f} Train Acc: {:.4f} | Val Loss: {:.4f} Val Acc: {:.4f}'
                .format(train_losses, train_accs, val_losses, val_accs))
    print("saving model ...")
    torch.save(model.state_dict(), path)
    return path


def test_model(dataloader, model, PATH=None, mode='test'):
    criterion = nn.CrossEntropyLoss()            
    loss = 0.0
    acc = 0    
    logits = []
    
    if PATH:
        validate_path(PATH)
        print(f"loading saved model - {PATH}....")
        model.load_state_dict(torch.load(PATH))         
        
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.eval()
   
    for inp, tar in dataloader:            
        inputs = inp.to(conf.device)
        target = tar.to(conf.device)                            
        optimizer.zero_grad()
        pred = model(inputs)                        
        loss = criterion(pred, target) 
        _, preds = torch.max(pred, 1)                                   
        loss += loss.item() * inputs.size(0)
        acc += torch.sum(preds == target.data) 
        logits.append(pred.cpu().detach())       
        
    if mode == 'train':
        return loss, acc
    else:  
        print('Test Loss: {}, Acc: {}'.format(loss/len(dataloader.dataset), acc/len(dataloader.dataset)))                     
        return torch.cat(logits, dim=0) 


def train_attacker(model, data):
    path = f'./weights/NN_attack_2layer.pt'
    #loss_func = SupervisedContrastiveLoss()       
    loss_bce = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)          
    batch = 200    
    trainloader = DataLoader(data, shuffle=True, batch_size=batch)    
    print(len(trainloader))    
    accur, losses = [], []
    model.train()    
    for epoch in range(700):
        acc = 0
        for data, label in trainloader:                    
            out = model(data)                        
            loss_ = loss_bce(out, label.reshape(-1, 1).type(torch.FloatTensor))
            acc += torch.sum(out.reshape(-1).round() == label)  
            optimizer.zero_grad()
            loss_.backward()            
            optimizer.step()               
        losses.append(loss_.cpu().detach().numpy())
        accur.append(acc.cpu().detach().numpy()/len(trainloader.dataset))
        if epoch % 10 == 0:    
            print("epoch = %4d  | loss = %10.4f, acc = %10.4f" % (epoch, loss_, acc/len(trainloader.dataset)))
    plot_loss(losses, accur)
    print("saving model ...")
    torch.save(model.state_dict(), path)


def test_attacker(model, data):
    path = f'./weights/NN_attack_2layer.pt'    
    if model:
        validate_path(path)
        print(f"loading saved attack model - {path}....")
        model.load_state_dict(torch.load(path)) 
    
    loss_bce = torch.nn.BCELoss()     
    batch = 100    
    testloader = DataLoader(data, shuffle=True, batch_size=batch)        
    acc_1 = 0
    acc_2 = 0

    model.eval()
    for data, label in testloader:  
        out = model(data) 
        loss_ = loss_bce(out, label.reshape(-1, 1).type(torch.FloatTensor))        
        for pred, tru in zip(out.reshape(-1), label):
            acc_1 += 1 if pred > 0.5 else 0
            acc_2 += 1 if pred.round() == tru else 0
        #acc += torch.sum(out.reshape(-1).round() == label) 
        #print(out.reshape(-1).round(), label)
    print("Test loss = %10.4f, Test acc_1 = %10.4f, acc_2 = %10.4f" % (loss_, acc_1/len(testloader.dataset), acc_2/len(testloader.dataset)))                              
            
    