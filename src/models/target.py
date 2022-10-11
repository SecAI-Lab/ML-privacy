import torchvision
import conf
from actions import train_model
from models import BaseModel
import torch
import torch.nn as nn

class DenseNet(BaseModel):
    def __init__(self, num_classes) -> None:
        super().__init__(num_classes)                        
        self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)       
        
        
    def train(self, dataloader):                
        model = self.fine_tune()
        model = model.to(conf.device)
        train_model(dataloader, model, 'target')


class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x