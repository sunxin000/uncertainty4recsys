import numpy as np
import matplotlib.pyplot as plt

train_matrix = np.loadtxt(f"./data/coat/train.ascii", dtype=int)
train_propensity = np.loadtxt(f"./data/coat/propensities.ascii", dtype=float)

user, item = np.where(train_matrix)
target = train_matrix[user, item]
propensity = train_propensity[user, item]

plt.hist(propensity, bins=20)

plt.title("Propensity Analyze")
plt.xlabel('Propensity')
plt.ylabel('Amount')

plt.savefig('propensity_analyze.jpg')