import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import validate_path
import conf


def train_model(dataloader, model, path, mode='target', rnn=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainloader = dataloader.target_trainloader
    testloader = dataloader.target_valloader

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

        print('Train Loss: {:.4f} Train Acc: {:.4f} | Val Loss: {:.4f} Val Acc: {:.4f}'
              .format(train_losses, train_accs, val_losses, val_accs))
    print("saving model ...")
    torch.save(model.state_dict(), path)


def test_model(dataloader, model, path=None, mode='test', rnn=True):
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    acc = 0
    logits, tru_preds = [], []

    if path:
        validate_path(path)
        print(f"loading saved model - {path}....")
        model.load_state_dict(torch.load(path))

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

    if mode == 'train':
        return loss, acc
    else:
        print('Test Loss: {}, Acc: {}'.format(
            loss/len(dataloader.dataset), acc / len(dataloader.dataset)))
        return torch.cat(logits, dim=0), torch.cat(tru_preds, dim=0)
