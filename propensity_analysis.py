import numpy as np
import matplotlib.pyplot as plt
import torch
# train_matrix = np.loadtxt(f"./data/coat/train.ascii", dtype=int)
# train_propensity = np.loadtxt(f"./data/coat/propensities.ascii", dtype=float)

# user, item = np.where(train_matrix)
# target = train_matrix[user, item]
# propensity = train_propensity[user, item]
# torch.save(propensity, 'data/propensity/coat_origin.pt')
# np.savetxt('data/propensity/coat_origin.txt', propensity)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10)
args = parser.parse_args()
epoch = args.epoch
path = f'data/propensity/raw/coat_epoch_{epoch}_1_dropout_0.2_label_smoothing_0.0.pt'
propensity = torch.load(path)
propensity = propensity[:len(propensity)//2]
plt.hist(propensity, bins=20)

# plt.title(path)
plt.xlabel('Propensity')
plt.ylabel('Amount')

plt.savefig(f'{path}_positive.jpg')