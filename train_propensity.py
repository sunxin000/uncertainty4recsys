from torchmetrics.functional.classification.accuracy import accuracy
from utils.dataset import Observe

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np


from utils.metrics import metrics
from model.neumf import NeuMF


embedding_size = 64
batch_size = 1024
data = "yahoo"
epoch = 500 if data == "coat" else 50
sample_ratio = 4
train = Observe(data, True, sample_ratio=sample_ratio)
test = Observe(data, False, sample_ratio=sample_ratio)
lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NeuMF(train.user_num, train.item_num, embedding_size, embedding_size, [32, 32, 32, 32])

print(len(train))
train_size  = int(0.9 * len(train))
validation_size = len(train) - train_size
train, validation = random_split(train, [train_size, validation_size])

train_loader = DataLoader(dataset=train, batch_size=1024, shuffle=True, num_workers=8)
val_loader = DataLoader(dataset=validation, batch_size=1024, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test, batch_size=1024, shuffle=True, num_workers=8)

#! dont knwo whether the testset has unknown user
#? no 
model = model.to(device)
loss_func = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= 0.001)

best_hr = 0
batches = len(train_loader)

# patient = 20
start_checking_epoch = 10

print(f"the num of batches is {batches}")
for epoch in range(1, epoch + 1):

    best_acc = 0
    model.train()
    # loss_tmp = 0
    acc = []
    for user, item, label in train_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_func(prediction, label.float())
        loss.backward()
        optimizer.step()

        
        acc.append(accuracy(prediction, label).cpu().numpy())


        # loss_tmp += loss.item()
    cur_acc = np.mean(acc)
    print(cur_acc)



    #! for early stopping 

    # loss_tmp /= batches
    # print(f"Epoch {epoch}: loss {loss_tmp}")
    model.eval()


    _, _, acc = metrics(model, val_loader, 2, device)

    if epoch > start_checking_epoch and acc > best_acc:
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state,  f"saved_propensity_model/neumf_propensity_{sample_ratio}_{data}.ckpt")
    print(f"acc {acc:.3f}")



