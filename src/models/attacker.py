from sklearn import ensemble, linear_model, neural_network, neighbors
import joblib
import numpy as np
import torch
import torch.nn as nn
from utils import utils
import os
import logging

logging.basicConfig(level=logging.INFO)


class EnsembleAttacker:
    def __init__(self, attackers_path) -> None:
        self.rf = ensemble.RandomForestClassifier(n_jobs=1)
        self.rf.set_params(
            n_estimators=500, criterion='gini', max_features='sqrt')

        self.lr = linear_model.LogisticRegression(solver='lbfgs', n_jobs=1)
        self.knn = neighbors.KNeighborsClassifier(n_jobs=1)
        self.mlp = neural_network.MLPClassifier()

        self.models = {
            'RF': (self.rf, attackers_path['rf_path']),
            'LR': (self.lr, attackers_path['lr_path']),
            'KNN': (self.knn, attackers_path['knn_path']),
            'MLP': (self.mlp, attackers_path['mlp_path'])
        }

    def train(self, train_data):
        x, y = utils.split_data(train_data)
        for name, (model, path) in self.models.items():
            logging.info(f'Training attacker {name}...')
            model.fit(x, y)
            joblib.dump(model, path)
            logging.info(f'Saved {name} attacker to {path}')

    def test(self, test_data):
        x, y = utils.split_data(test_data)
        for name, (model, path) in self.models.items():
            logging.info(f'Testing {name} attack model')
            if os.path.exists(path):
                model = joblib.load(path)
            else:
                model = model
            preds = model.predict(x)
            tru_preds = 0
            for t, p in zip(y, preds):
                tru_preds += 1 if (p == t) else 0
            print("Acc of attacker {}: {}, {}/{}".format(name, tru_preds /
                                                         len(preds), tru_preds, len(preds)))


class NNAttack(nn.Module):
    pass
# def __init__(self, n_classes) -> None:
#     super().__init__()
    # 	self.fc1 = nn.Linear(n_classes, 32)
#     self.fc2 = nn.Linear(32, 64)
#     self.fc3 = nn.Linear(32, 1)

    # def forward(self, x):
#     x = torch.relu(self.fc1(x))
#     #x = self.dropout(x)
    # 	#x = torch.relu(self.fc2(x))

#     x = torch.sigmoid(self.fc3(x))
#     return x


"""

class ContrastiveLoss(nn.Module):  

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


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):        
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):        
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        #print(targets.shape)
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        #print(mask_similar_class.shape)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

"""
