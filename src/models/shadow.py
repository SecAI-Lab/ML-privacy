import conf
import torchvision
from actions import train_model
from models.target import DenseNet
from models import BaseModel


class ShadowModel(BaseModel):
    def __init__(self, num_classes, is_whitebox=True) -> None:
        super().__init__(num_classes)
        self.is_whitebox = is_whitebox
        self._init_model()

    def _init_model(self):
        if self.is_whitebox:
            net = DenseNet(num_classes=self.classes)
            self.model = net.model
        else:
            self.model = torchvision.models.resnet101(
                weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)

    def get_model(self):
        if self.is_whitebox:
            return self.update_out_layer('resnet')
        else:
            return self.update_out_layer()

    def train(self, dataloader, path):
        model = self.get_model()
        model = model.to(conf.device)
        train_model(dataloader, model, path, 'shadow', rnn=False)
