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
path = 'data/propensity/raw/coat_epoch_10_-1.pt'
propensity = torch.load(path)
plt.hist(propensity, bins=20)

plt.title(path)
plt.xlabel('Propensity')
plt.ylabel('Amount')

plt.savefig(f'{path}.jpg')