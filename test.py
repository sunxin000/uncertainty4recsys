import torch
# import numpy as np

# propensity_raw = torch.load('data/propensity/raw/coat_epoch_10_1_dropout_0.2_label_smoothing_0.0.pt')
# propensity_label_smoothing = torch.load('data/propensity/raw/coat_epoch_10_1_dropout_0.2_label_smoothing_0.1.pt')

# propensity_diff = np.abs(propensity_raw - propensity_label_smoothing)

# index_sort = np.argsort(-propensity_diff)

# print(index_sort[:100])

propensity = torch.load('data/propensity/raw/1_label_smoothing_10_20.pt')
flag = propensity > 1
print(propensity[flag])
print(flag.any())