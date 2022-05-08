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
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument('--mlp_layers',
                        nargs='*',
                        type=int,
                        default=[64, 32, 16])
    parser.add_argument("--sample_ratio", type=int, default=-1)
    parser.add_argument("--n_model", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    embedding_size = args.embedding_size
    data = args.dataset
    sample_ratio = args.sample_ratio
    n_model = args.n_model
    batch_size = 1024
    train = Observe(data, True, sample_ratio=sample_ratio)
    user_num = train.user_num
    item_num = train.item_num
    train_loader = DataLoader(dataset=train,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)
    paths = [
        f'saved_propensity_model/coat/ensemble/neumf_-1_{args.epoch}_{i}_with_seed.ckpt'
        for i in range(1, n_model + 1)
    ]
    mlp_layer = args.mlp_layers
    model = NeuMF(user_num, item_num, embedding_size, embedding_size,
                  mlp_layer)
    len_preds = len(train)
    predictions = torch.zeros(len_preds, )
    model.eval()
    with torch.no_grad():
        for path in paths:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['net'])
            for index, (user, item, label) in enumerate(train_loader):
                pred = model(user, item)
                pred = pred / n_model
                predictions[index * batch_size:min(
                    (index + 1) * batch_size, len(predictions))] += pred

    torch.save(predictions.detach(),
               f"data/propensity/ensemble/{sample_ratio}_{args.epoch}.pt")
    predictions = np.array(predictions.detach())
    np.savetxt(f"data/propensity/ensemble/{sample_ratio}_{args.epoch}.txt",
               predictions)


if __name__ == "__main__":
    main()