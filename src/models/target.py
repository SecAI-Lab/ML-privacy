import torchvision
import torch.nn as nn
import torch
import conf
import torch.optim as optim


class DenseNet:
    def __init__(self, num_classes) -> None:                
        self.model = torchvision.models.densenet121(pretrained=True)       
        self.classes = num_classes
            
    def fine_tune(self):            
        cnn = self.model        
        cnn.classifier = nn.Linear(cnn.classifier.in_features, self.classes)
        return cnn
    
    def train(self, trainloader):        
        print("Training..")
        model = self.fine_tune()
        model = model.to(conf.device)
        criterion = nn.CrossEntropyLoss()        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            for epoch in range(conf.epochs):
                for inputs, target in trainloader:
                    inputs = inputs.to(conf.device)
                    target = target.to(conf.device)
                    
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = model(inputs)                        
                        loss = criterion(pred, target)                        
                        
                        _, preds = torch.max(pred, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == target.data)

                epoch_loss = running_loss / len(inputs)
                epoch_acc = running_corrects.double() / len(inputs)

                print('Epoch: {} -- {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))







