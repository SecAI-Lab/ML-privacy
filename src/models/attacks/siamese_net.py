from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, auc, roc_curve
import scikitplot as skplt


class SiamAttacker:
    def __init__(self, path, n_classes) -> None:
        self.net = SiameseNetwork(n_classes)
        self.optim = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = ContrastiveLoss()
        self.model_path = path

    def train(self, train_data):
        losses = []
        epochs = 50
        for epoch in range(epochs):
            loss = 0
            for vec1, vec2, label in train_data:
                self.optim.zero_grad()
                output1, output2 = self.net(vec1, vec2)
                loss_contrastive = self.criterion(output1, output2, label)
                loss += loss_contrastive.item()
                loss_contrastive.backward()
                self.optim.step()

            losses.append(loss/len(train_data))
            print(
                f"Epoch number {epoch}\tCurrent loss {loss/len(train_data)}\n")

        plt.plot(range(epochs), losses)
        plt.savefig('train_loss.png')

        print('Saving model in ', self.model_path)
        torch.save(self.net.state_dict(), self.model_path)

    def test(self, test_data):
        self.net.load_state_dict(torch.load(self.model_path))
        acc, f1, roc_auc = 0, 0, 0
        all_fpr, all_tpr = 0, 0
        for vec1, vec2, label in test_data:
            self.optim.zero_grad()
            output1, output2 = self.net(vec1, vec2)
            euclid_dist = F.pairwise_distance(output1, output2)
            min_value = torch.min(euclid_dist)
            max_value = torch.max(euclid_dist)
            thrsh = (min_value + max_value)/2
            normalized = (euclid_dist - min_value) / \
                (max_value - min_value)
            preds = torch.where(
                normalized < thrsh, torch.tensor(1), torch.tensor(0))
            acc += torch.sum(preds == label)
            f1 += f1_score(label, preds, average='weighted')
            fpr, tpr, t = roc_curve(label, preds)
            roc_auc += auc(fpr, tpr)
            all_fpr += fpr
            all_tpr += tpr

        print('Acc - ', acc/len(test_data.dataset))
        print('F1 score - ', f1/len(test_data))
        print('AUC score - ', roc_auc/len(test_data))
        # plt.plot(all_fpr/len(test_data), all_tpr/len(test_data), label="Attack AUC = " +
        #          str(roc_auc/len(test_data)))
        # plt.legend(loc=4)
        # plt.savefig('roc.png')


class SiameseNetwork(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_classes, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 2)
        )

    def forward_once(self, x):
        output = self.fc(x)
        return output

    def forward(self, vec1, vec2):
        out1 = self.forward_once(vec1)
        out2 = self.forward_once(vec2)
        return out1, out2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclid_dist = F.pairwise_distance(
            out1, out2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclid_dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclid_dist, min=0.0), 2))
        return loss_contrastive
