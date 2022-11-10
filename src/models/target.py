import torchvision
import conf
from actions import train_model
from models import BaseModel
import torch.nn as nn

class DenseNet(BaseModel):
    """CNN model for target"""
    def __init__(self, num_classes) -> None:
        super().__init__(num_classes)                        
        self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)       
                
    def train(self, dataloader):                
        model = self.fine_tune()
        model = model.to(conf.device)
        train_model(dataloader, model, conf.cifar_target_path, 'target', rnn=False)


class GRU(nn.Module):
    """RNN model for target"""
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) 
        x, _ = self.gru(x, h_0)  
        h_t = x[:,-1,:] 
        self.dropout(h_t)
        logit = self.out(h_t)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()




"""
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
"""