import conf
from actions import train_model
from models.target import DenseNet
from models import BaseModel

class ShadowModel(BaseModel):
    def __init__(self, num_classes, model_type='densenet', is_whitebox=True) -> None:                        
        super().__init__(num_classes)
        self.model_type = model_type
        self.model = self.get_model()        
        self.is_whitebox = is_whitebox
            
    def get_model(self):
        if self.model_type == 'densenet':
            target = DenseNet(num_classes=self.classes) 
            model = target.model
        else:
            """To be implemented"""
            pass
        return model
    
    def get_white_box(self):
        return self.fine_tune()
    
    def get_black_box(self):
        #TODO: Extending model for black box
        pass

    def train(self, dataloader):                
        if self.is_whitebox:
            model = self.get_white_box()
        
        model = model.to(conf.device)
        train_model(dataloader, model, 'shadow')