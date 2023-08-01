from abc import ABC
import torch.nn as nn
from actions import test_model
import conf

class BaseModel(ABC):
    def __init__(self, num_classes) -> None:
        self.model = None
        self.classes = num_classes

    def update_out_layer(self, model_name='densenet'):
        model = self.model                        
        if model_name == 'densenet':            
            model.classifier = nn.Linear(model.classifier.in_features, self.classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, self.classes)
        return model

    def test(self, model_path, dataloader):        
        if 'resnet' in model_path:
            model = self.update_out_layer('resnet')
        else:            
            model = self.update_out_layer()
        model = model.to(conf.device)
        probs = test_model(dataloader, model, model_path, rnn=False)
        return probs
   
    
    