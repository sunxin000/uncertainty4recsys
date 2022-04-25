from sqlalchemy import true
import torch
from torch.utils.data import random_split, DataLoader
from model.neumf import NeuMF
from utils.dataset import Observe
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coat")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--sample_ratio", type=int, default=0)
    parser.add_argument("--dropout", aciton='store_ture')
    parser.add_argument("--n_inferences", type=int, default=10)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    embedding_size = args.embedding_size
    mlp_dim = args.mlp_dim
    data = args.dataset
    sample_ratio = args.sample_ratio
    dropout = args.dropout
    n_inferences = args.n_inferences
    n_layer = 4
    batch_size = 1024
    path = f'saved_propensity_model/neumf_propensity_{sample_ratio}_{data}_{embedding_size}_{mlp_dim}.ckpt'


    train = Observe(data, True, 0)
    user_num = train.user_num
    item_num = train.item_num
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlp_layer = [mlp_dim] * n_layer
    model = NeuMF(user_num, item_num,  embedding_size, embedding_size, mlp_layer)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    
    
    len_preds = 311704 if data == 'yahoo' else 6960
    predictions = torch.zeros(len_preds,)
    if not dropout:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        for i in range(n_inferences):
            for index, (user, item, label) in enumerate(train_loader):
                pred = model(user, item)
                pred = pred / n_inferences
                predictions[index * batch_size: min((index + 1) * batch_size, len(predictions))] += pred

    dir = 'dropout' if dropout else 'raw'
    torch.save(predictions.detach(), f"data/propensity/{dir}/neumf_propensity_{sample_ratio}_{data}_{embedding_size}_{mlp_dim}.pt")
    predictions = np.array(predictions.detach())
    np.savetxt(f"data/propensity/{dir}/neumf_propensity_{sample_ratio}_{data}_{embedding_size}_{mlp_dim}.txt", predictions)

if __name__ == "__main__":
    main()