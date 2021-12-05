import torch.nn as nn
import torch
from .layers import MLPLayers
from torch.nn.init import normal_

class NeuMF(nn.Module):
    def __init__(
        self, num_users, num_items, mf_emb_size, mlp_emb_size, mlp_hidden_size, dropout=0., mf_train=True, mlp_train=True
        ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_emb_size = mf_emb_size
        self.mlp_emb_size = mlp_emb_size
        self.dropout = dropout
        self.mlp_hidden_size = mlp_hidden_size
        self.mf_train = mf_train
        self.mlp_train = mlp_train

        self.user_mf_embedding = nn.Embedding(self.num_users, self.mf_emb_size)
        self.item_mf_embedding = nn.Embedding(self.num_items, self.mf_emb_size)
        self.user_mlp_embedding = nn.Embedding(self.num_users, self.mlp_emb_size)
        self.item_mlp_embedding = nn.Embedding(self.num_items, self.mlp_emb_size)

        self.mlp_layers = MLPLayers([2 * self.mlp_emb_size] + self.mlp_hidden_size, self.dropout, init_method= 'xavier')

        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_emb_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_emb_size, 1)
        else:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.apply(self._init_weights)
    
    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')
        return output.squeeze(-1)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)
        



        
    
        






