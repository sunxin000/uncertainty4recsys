# import torch
# # import numpy as np

# # propensity_raw = torch.load('data/propensity/raw/coat_epoch_10_1_dropout_0.2_label_smoothing_0.0.pt')
# # propensity_label_smoothing = torch.load('data/propensity/raw/coat_epoch_10_1_dropout_0.2_label_smoothing_0.1.pt')

# # propensity_diff = np.abs(propensity_raw - propensity_label_smoothing)

# # index_sort = np.argsort(-propensity_diff)

# # print(index_sort[:100])

# propensity = torch.load('data/propensity/raw/1_label_smoothing_10_20.pt')
# flag = propensity > 1
# print(propensity[flag])
# print(flag.any())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset import Observe, ObservedData
import time
import tqdm
data = 'yahoo'
propensity = torch.load('propensity/ls/yahoo.pt')
# propensity = torch.tensor(propensity).cuda()
train_il = ObservedData(data,
                        train=True,
                        implicit=True,
                        propensity=propensity)

train_loader = DataLoader(dataset=train_il,
                            batch_size=1024,
                            shuffle=True,
                            # num_workers=8,
                            # pin_memory=True,)
)
device = torch.device('cuda:0')
start = time.time()
for epoch in tqdm.tqdm(range(10)):
    for user, item, label, p in train_loader:
        # user = user.cuda(non_blocking=True)
        # item = item.cuda(non_blocking=True)
        # label = label.cuda(non_blocking=True)
        # p = p.cuda(non_blocking=True)
        pass
end = time.time()

print(end-start)