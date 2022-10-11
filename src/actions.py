from operator import not_
from sys import flags
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import conf
from utils.plotter import *
from utils.utils import validate_path
from models.attack import ContrastiveLoss

def train_model(dataloader, model, mode='target'):
    criterion = nn.CrossEntropyLoss()        
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    trainloader = dataloader.target_trainloader
    testloader = dataloader.target_valloader
    losses = {'train': [], 'val': []}
    path = f'./weights/{mode}_stl10.pt'
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
    label_preds = []
    labels, logits = [], []
    
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
        label_preds.append(preds.cpu().detach())           
        labels.append(target.data.cpu().detach())        
        
    if mode == 'train':
        return loss, acc
    else:  
        print('Test Loss: {}, Acc: {}'.format(loss/len(dataloader.dataset), acc/len(dataloader.dataset)))                     
        return torch.cat(labels, dim=0), torch.cat(logits, dim=0) 


def train_attacker(model, pos_sample, pos_sample_, neg_sample):
    loss_func = ContrastiveLoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    ep_log_interval = 4
    #print(pos_sample.shape)
    # pos_sample = torch.transpose(torch.from_numpy(pos_sample), (1, 0, 2))
    # neg_sample = torch.transpose(torch.from_numpy(neg_sample), (1, 0, 2))    
    print(pos_sample.shape)

    model.train()    
    for epoch in range(40):        
        p_loss, n_loss = 0, 0
        #for x1, x2 in zip(in_train, not_in_train):            
        x1, x2 = model(pos_sample, pos_sample_)            
        y1, y2 = model(pos_sample, neg_sample)            
        optimizer.zero_grad()              
        loss_pos = loss_func(x1, x2, flag=0)        
        loss_neg = loss_func(y1, y2, flag=1)        
        p_loss += loss_pos.item() 
        n_loss += loss_neg.item() 
        loss_pos.backward()        
        loss_neg.backward()        
        optimizer.step() 
        #print('pos', loss_pos.item())
        #print('neg', loss_neg.item())    
        if epoch % ep_log_interval == 0:
            print("epoch = %4d  | pos_loss = %10.4f, neg_loss = %10.4f," % (epoch, p_loss, n_loss))
    print('member', torch.max(y1, dim=1))
    print('non_member', torch.max(y2, dim=1))
    print("Done ") 


def test_attacker(model, data):
    loss = nn.MSELoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    inpt = torch.FloatTensor(np.array([x[0] for x in data]))
    labels = torch.FloatTensor(np.array([y[1] for y in data]))
    model.eval()
    preds = model(inpt)
    optimizer.zero_grad()
    loss_val = loss(preds, labels)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds =roc_curve(labels, preds.cpu().detach().numpy())
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)    
    print(thresholds)
    thresh = 0.5    
    bin_preds = []
    # for pred, tru in zip(preds, labels):
    #     print(pred, tru)
        #if Pred > thresh:
        #    bin_preds.append(1)
        #else:
        #    bin_preds.append(0)

    correct = 0
    #for pred, target in zip(bin_preds, labels):
    #    if pred == target:
    #        correct += 1
    #print('Accuracy:', correct/len(labels))
    

