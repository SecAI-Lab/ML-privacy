from abc import ABC
import torch.nn as nn
from actions import test_model
import conf

class BaseModel(ABC):
    def __init__(self, num_classes) -> None:
        self.model = None
        self.classes = num_classes

    def fine_tune(self):
        model = self.model        
        model.classifier = nn.Linear(model.classifier.in_features, self.classes)
        return model

    def test(self, model_path, dataloader):
        model = self.fine_tune()
        model = model.to(conf.device)
        probs = test_model(dataloader, model, model_path, rnn=False)
        return probs
   
    
    