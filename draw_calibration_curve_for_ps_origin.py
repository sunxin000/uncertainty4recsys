import numpy as np
from utils.dataset import Observe
from sklearn.calibration import calibration_curve, CalibrationDisplay
from matplotlib import pyplot as plt
sample_ratio = -1
dataset = Observe(train='train', sample_ratio=sample_ratio)
user = dataset.user
item = dataset.item
label = dataset.target

propensity_matrix = np.loadtxt(f"./data/coat/propensities.ascii", dtype=float)

preds = propensity_matrix[user, item]

assert preds.shape == label.shape
n_bins = 10
disp = CalibrationDisplay.from_predictions(label, preds, n_bins=n_bins)# , strategy='quantile')
# title = f'given_propensity_score_quantile_{n_bins}_{sample_ratio}'
# plt.title(title)
plt.savefig(f"pic/origin.jpg")