from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy  as np

@dataclass
class WhiteBoxDataloader:
    target_trainloader: DataLoader
    target_valloader: DataLoader
    shadow_trainloader: DataLoader
    shadow_valloader: DataLoader
    #testloader: DataLoader
    

@dataclass
class AttackData:
    test_preds: np.array
    test_labels: np.array
    train_preds: np.array    
    train_labels: np.array
    