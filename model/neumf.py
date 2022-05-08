import torch.nn as nn
import torch
from .layers import MLPLayers
from torch.nn.init import normal_

class NeuMF(nn.Module):
    def __init__(
        self, num_users, num_items, mf_emb_size, mlp_emb_size, mlp_hidden_size, dropout=0.,
        ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_emb_size = mf_emb_size
        self.mlp_emb_size = mlp_emb_size
        self.dropout = dropout
        self.mlp_hidden_size = mlp_hidden_size


        self.user_mf_embedding = nn.Embedding(self.num_users, self.mf_emb_size)
        self.item_mf_embedding = nn.Embedding(self.num_items, self.mf_emb_size)
        self.user_mlp_embedding = nn.Embedding(self.num_users, self.mlp_emb_size)
        self.item_mlp_embedding = nn.Embedding(self.num_items, self.mlp_emb_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.mlp_layers = MLPLayers([2 * self.mlp_emb_size] + self.mlp_hidden_size, self.dropout, init_method= 'xavier')
        self.predict_layer = nn.Linear(self.mf_emb_size + self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.apply(self._init_weights)
    
    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        user_mf_e = self.dropout_layer(user_mf_e)
        item_mf_e = self.dropout_layer(item_mf_e)
        user_mlp_e = self.dropout_layer(user_mlp_e)
        item_mlp_e = self.dropout_layer(item_mlp_e)
        
        mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        return output.squeeze(-1)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)
        



        
    
        






