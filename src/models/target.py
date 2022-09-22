import torchvision
import torch.nn as nn
import torch
import conf
import torch.optim as optim
from tqdm import tqdm


class DenseNet:
    def __init__(self, num_classes) -> None:                
        self.model = torchvision.models.densenet121(pretrained=True)       
        self.classes = num_classes
            
    def fine_tune(self):            
        cnn = self.model        
        cnn.classifier = nn.Linear(cnn.classifier.in_features, self.classes)
        return cnn
    
    def train(self, trainloader, testloader):                
        model = self.fine_tune()
        model = model.to(conf.device)
        criterion = nn.CrossEntropyLoss()        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
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

                if i % 100 == 0:
                    val_loss = 0.0
                    val_acc = 0
                    model.eval()
                    for inp, tar in testloader:
                        inputs = inp.to(conf.device)
                        target = tar.to(conf.device)                
                        optimizer.zero_grad()
                        pred = model(inputs)                        
                        loss = criterion(pred, target) 
                        _, preds = torch.max(pred, 1)                    
                        
                        val_loss += loss.item() * inputs.size(0)
                        val_acc += torch.sum(preds == target.data)
                
                train_loss += loss.item() * inputs.size(0)
                train_acc += torch.sum(preds == target.data)
                
            train_losses = train_loss / len(trainloader.dataset)
            train_accs = train_acc.double() / len(trainloader.dataset)

            val_losses = val_loss / len(testloader.dataset)
            val_accs = val_acc.double() / len(testloader.dataset)

            print('Train Loss: {:.4f} Train Acc: {:.4f} | Val Loss: {:.4f} Val Acc: {:.4f}'
                  .format(train_losses, train_accs, val_losses, val_accs))
            







