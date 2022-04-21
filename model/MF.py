import torch.nn as nn
import torch
from torch.nn.init import normal_

class MF(nn.Module):
    def __init__(
        self,num_users, num_items,emb_size,dropout=0.
        ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_size = emb_size
        self.dropout = dropout

        self.user_embedding = nn.Embedding(self.num_users, self.emb_size)
        self.item_embedding = nn.Embedding(self.num_items, self.emb_size)
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)

        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        self.user_bias.weight.data.fill_(0.)
        self.item_bias.weight.data.fill_(0.)
        self.global_bias = nn.Parameter(torch.FloatTensor([0.]), requires_grad=True)
    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        pred = torch.sum(user_e * item_e, 1, keepdim=True) 
        pred += user_bias + item_bias + self.global_bias
        pred = self.sigmoid(pred)
        return pred.squeeze()

