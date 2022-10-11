from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import numpy as np
import torch
import torch.nn as nn
import os


class DistanceAttack(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_classes, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def feed(self, x):        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def forward(self, x1, x2):                
        x1 = self.feed(x1)
        #print('learned member:', x1)
        x2 = self.feed(x2)
        #print('non-member:', x2)
        return x1, x2


class RandomForest:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(n_jobs=1)
        self.model.set_params(n_estimators=500, criterion='gini', max_features='sqrt')
        self.path = './weights/RF_attack_with_shadow.joblib'

    def split_data(self, data):
        x = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        return x, y
    
    def train(self, train_data):
        print("Training attacker...")        
        x, y = self.split_data(train_data)                        
        self.model.fit(x, y)     
        print('Saving RF attack model')
        joblib.dump(self.model, self.path)  
    
    def test(self, test_data):        
        print("Testing attacker on target outputs...")
        x, y = self.split_data(test_data)
        if os.path.exists(self.path):
            print('Loading RF attack model')
            model = joblib.load(self.path)    
        else:
            model = self.model        
        preds = model.predict(x)
        tru_preds = 0                
        for t, p in zip(y, preds):            
            tru_preds += 1 if (p == t) else 0        
        print("Acc of attacker: {}, {}/{}".format(tru_preds/len(preds), tru_preds, len(preds)))
       
        

class ContrastiveLoss(nn.Module):
  """Contrastive Loss function for Contrastive learning"""

  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__() 
    self.m = m 

  def forward(self, y1, y2, flag):
    # flag = 0 means y1 and y2 are supposed to be same
    # flag = 1 means y1 and y2 are supposed to be different
    euc_dist = nn.functional.pairwise_distance(y1, y2)

    if flag == 0:
      return torch.mean(torch.pow(euc_dist, 2))  
    else: 
      delta = self.m - euc_dist  
      delta = torch.clamp(delta, min=0.0, max=None)
      return torch.mean(torch.pow(delta, 2))  

    

