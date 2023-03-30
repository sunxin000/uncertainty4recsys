import numpy as np
import torch
from netcal.metrics import ECE, MCE
from torch import nn, optim
from torch.nn import functional as F


class ModelWithPlatt(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.a = nn.Parameter(torch.FloatTensor([0.6],).cuda())
        self.b = nn.Parameter(torch.FloatTensor([0.]).cuda())


    def forward(self, input):
        logits = self.model(input)
        return self.platt_scale(logits)
    
    def platt_scale(self, logits):
        return torch.mul(logits, self.a) + self.b
    

    def set_ab(self, valid_loader):
        nll_criterion = nn.BCEWithLogitsLoss()
        ece_criterion = ECE(bins=100).measure

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for user, item, labels in valid_loader:
                user = user.cuda()
                item = item.cuda()
                logits = self.model(user, item)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

            before_platt_nll = nll_criterion(logits, labels.float()).item()
            before_platt_ece = ece_criterion(np.asarray(torch.sigmoid(logits).cpu()), np.asarray(labels.cpu()))

            print(before_platt_nll)
            print(before_platt_ece)

            optimizer = optim.LBFGS([self.a, self.b], lr=0.01, max_iter=1000)

            def eval():
                optimizer.zero_grad()
                #TODO only platt scale for positive samples
                positive_logits = logits[labels==1]
                positive_labels = labels[labels==1]
                loss = nll_criterion(self.platt_scale(positive_logits), positive_labels.float())
                loss.backward()
                return loss
            optimizer.step(eval)

            after_platt_nll = nll_criterion(self.platt_scale(logits), labels.float()).item()
            after_platt_ece = ece_criterion(np.asarray(torch.sigmoid(self.platt_scale(logits)).cpu()), np.asarray(labels.cpu()))
            print(after_platt_nll)
            print(after_platt_ece)
            print(self.a.item(), self.b.item())
            return self, logits, labels





        
        