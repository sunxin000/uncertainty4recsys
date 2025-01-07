import torch
from torch.utils.data import random_split, DataLoader
from model.neumf import NeuMF
from utils.dataset import Observe
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.model_selection import KFold
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coat")
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument('--mlp_layers', nargs='*', type=int, default=[64, 32, 16])
    parser.add_argument("--sample_ratio", type=int, default=-1)
    parser.add_argument("--dropout", action='store_true')
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--n_inferences", type=int, default=1)
    parser.add_argument("--n_flod", default=5)
    parser.add_argument('--dir')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    embedding_size = args.embedding_size
    data = args.dataset
    sample_ratio = args.sample_ratio
    dropout = args.dropout
    n_inferences = args.n_inferences
    batch_size = 1024
    path = '/data/sunxin/uncertainty4recsys/propensity/saved_model/yahoo_ls_neumf_1_20_0_ls_0.1.ckpt'

    train = Observe(data, True, sample_ratio=sample_ratio)
    user_num = train.user_num
    item_num = train.item_num
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlp_layer = args.mlp_layers
    model = NeuMF(user_num, item_num,  embedding_size, embedding_size, mlp_layer, dropout=args.dropout_rate)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    
    
    len_preds = len(train)
    predictions = torch.zeros(len_preds,)
    model.eval()
    if dropout:
        model.dropout_layer.train()

    with torch.no_grad():
        for i in range(n_inferences):
            for index, (user, item, label) in enumerate(train_loader):
                pred = model(user, item)
                pred = pred / n_inferences
                predictions[index * batch_size: min((index + 1) * batch_size, len(predictions))] += pred

    dir = args.dir
    torch.save(predictions.detach(), f"propensity/{dir}/{data}_{sample_ratio}_{n_inferences}.pt")
    predictions = np.array(predictions.detach())
    np.savetxt(f"propensity/{dir}/{data}_{sample_ratio}.txt", predictions)

if __name__ == "__main__":
    main()