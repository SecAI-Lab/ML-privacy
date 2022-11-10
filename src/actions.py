import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import conf
from utils.plotter import *
from utils.utils import validate_path
import copy


def train_model(dataloader, model, path, mode='target', rnn=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainloader = dataloader.target_trainloader
    testloader = dataloader.target_valloader
    losses = {'train': [], 'val': []}

    if mode != 'target':
        print("Training shadow model...")
        trainloader = dataloader.shadow_trainloader
        testloader = dataloader.shadow_valloader

    for epoch in tqdm(range(conf.epochs)):
        print('Epoch {}'.format(epoch+1))
        model.train()
        train_loss = 0.0
        train_acc = 0

        for i, (inputs, target) in enumerate(trainloader):  # target

            if rnn:
                inputs, target = inputs.text.to(
                    conf.device), inputs.label.to(conf.device)
                target.data.sub_(1)
            else:
                inputs = inputs.to(conf.device)
                target = target.to(conf.device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, target)
            _, preds = torch.max(pred, 1)

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                val_loss, val_acc = test_model(
                    testloader, model, mode='train', rnn=rnn)

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(preds == target.data)

        train_losses = train_loss / len(trainloader.dataset)
        train_accs = train_acc.double() / len(trainloader.dataset)

        val_losses = val_loss / len(testloader.dataset)
        val_accs = val_acc.double() / len(testloader.dataset)
        # losses['train'].append(train_losses.cpu().detach().numpy())
        # losses['val'].append(val_losses.cpu().detach().numpy())
        # plot_loss(losses, mode)
        print('Train Loss: {:.4f} Train Acc: {:.4f} | Val Loss: {:.4f} Val Acc: {:.4f}'
              .format(train_losses, train_accs, val_losses, val_accs))
    print("saving model ...")
    torch.save(model.state_dict(), path)
    return path


def test_model(dataloader, model, PATH=None, mode='test', rnn=True):
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    acc = 0
    logits, tru_preds = [], []
    losses, outs = [], []

    if PATH:
        validate_path(PATH)
        print(f"loading saved model - {PATH}....")
        model.load_state_dict(torch.load(PATH))

    if mode == 'test':
        model.eval()

    for inp, tar in dataloader:   # tar
        if rnn:
            inp, tar = inp.text.to(conf.device), inp.label.to(conf.device)
            tar.data.sub_(1)
        else:
            inp = inp.to(conf.device)
            tar = tar.to(conf.device)

        pred = model(inp)
        loss = criterion(pred, tar)
        _, preds = torch.max(pred, 1)
        loss += loss.item()
        acc += torch.sum(preds == tar.data)
        logits.append(pred.cpu().detach())
        
        # extracting only correct predictions
        correct_preds = preds == tar.data
        indices = correct_preds.nonzero()
        correct_preds = torch.squeeze(pred[indices])
        tru_preds.append(correct_preds.cpu().detach())
        # if not rnn:
        #     loss.backward(retain_graph=True)
        #     for name, params in reversed(list(model.named_parameters())):
        #         if 'weight' in name:
        #             gradient = params.grad.clone()
        #             gradient = gradient.unsqueeze_(0)
        #             gradients.append(gradient.unsqueeze_(0))
        #             break

    # currently not used
    #gradients = torch.cat(gradients, dim=0)

    if mode == 'train':
        return loss, acc
    else:
        print('Test Loss: {}, Acc: {}'.format(
            loss/len(dataloader.dataset), acc / len(dataloader.dataset)))        
        return torch.cat(logits, dim=0), torch.cat(tru_preds, dim=0)


def train_attacker(model, train_data, val_data=None):
    path = './weights/NN_attack_2layer.pt'
    loss_func = nn.TripletMarginLoss(
        margin=1.0, p=2)  # SupervisedContrastiveLoss()
    loss_bce = torch.nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    batch = 100
    trainloader = DataLoader(train_data, shuffle=True, batch_size=batch)

    print(len(trainloader))
    accur, losses = [], []
    model.train()
    for epoch in range(20):
        acc = 0
        loss = 0
        for (logit, log_label), (grad, grad_label) in trainloader:
            out = model(logit, grad)
            labels = torch.cat([log_label, grad_label])
            # loss_ = loss_func(pos, pos_, neg)#label.reshape(-1, 1).type(torch.FloatTensor))
            loss_ = loss_bce(
                out, labels.reshape(-1, 1).type(torch.FloatTensor))

            acc += torch.sum(out.reshape(-1).round() ==
                             label) / len(trainloader.dataset)
            loss += loss_.item() / len(trainloader)

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
        # losses.append(loss_.cpu().detach().numpy())
        # accur.append(acc.cpu().detach().numpy()/len(trainloader.dataset))
        if val_data and epoch % 5 == 0:
            print("epoch = %4d  | loss = %10.4f, acc = %10.4f" %
                  (epoch, loss, acc))
            test_attacker(model, val_data, val=True)

    # plot_loss(losses, accur)
    print("saving model ...")
    torch.save(model.state_dict(), path)


def test_attacker(model, data, val=False):
    path = './weights/NN_attack_2layer.pt'
    if validate_path(path) and not val:
        print(f"loading saved attack model - {path}....")
        model.load_state_dict(torch.load(path))

    batch = 100
    testloader = DataLoader(data, shuffle=True, batch_size=batch)
    acc = 0

    model.eval()
    for data, label in testloader:
        out = model(data)
        acc += torch.sum(out.reshape(-1).round() == label)

    print("Test acc = %10.4f " % (acc/len(testloader.dataset)))
