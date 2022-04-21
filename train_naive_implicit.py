import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchmetrics.functional.classification.accuracy import accuracy
from model.MF import MF
import torch.autograd.profiler as profiler


from utils.metrics import metrics, dcg_at_k
from utils.dataset import ObservedData
torch.backends.cudnn.benchmark = True

# import argparse
# import torch.distributed as dist

# parser = argparse.ArgumentParser()
# parser.add_argument('--local_rank', default=-1, type=int,
#                     help='node rank for distributed training')
# args = parser.parse_args()
# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(args.local_rank)

lr = 1e-3
embedding_size = 64
batch_size = 1024
data = "coat"
epoch = 1000 if data == "coat" else 20

items_per_user = 16 if data == "coat" else 10

train = ObservedData(data, train=True, implicit=True)
# train = ObservedData(data, train=True, implicit=True)
test = ObservedData(data, train=False, implicit=True)
user_num, item_num = train.user_num, train.item_num

train_size = int(0.9 * len(train))
validation_size = len(train) - train_size
train, validation = random_split(train, [train_size, validation_size])

# train_sampler = torch.utils.data.distributed.DistributedSampler(train)

train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)
test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True)

# model = NeuMF(user_num, item_num, embedding_size, embedding_size, [32, 16, 8])
model = MF(user_num, item_num, embedding_size)
model = model.cuda()
#! dont knwo whether the testset has unknown user
#? no
# model = torch.nn.parallel.DistributedDataParallel(model,
#                                                   device_ids=[args.local_rank])

# loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr)  #! can adjust the weight_decay
loss_func = nn.BCELoss(reduction='none')
batches = len(train_loader)
print(f"the num of batches is {batches}")

# patient = 20
start_checking_epoch = 10

for epoch in range(1, epoch + 1):
# with profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    best_acc = 0
    model.train()
    # loss_tmp = 0
    acc = []
    for user, item, label in train_loader:
        user, item = user.cuda(), item.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_func(prediction, label.float())
        loss.sum().backward()

        # loss.backward()
        optimizer.step()

        acc.append(accuracy(prediction, label.long()).cpu().numpy())

        # loss_tmp += loss.item()
    cur_acc = np.mean(acc)
    print(f"train acc {cur_acc}")

    #! for early stopping

    # loss_tmp /= batches
    # print(f"Epoch {epoch}: loss {loss_tmp}")
    model.eval()

    PRECISION = []
    predictions = torch.empty(0)
    labels = torch.empty(0)

    for user, item, label in test_loader:
        user, item = user.cuda(), item.cuda()
        pred = model(user, item)
        predictions = torch.cat((predictions, pred.detach().cpu()))
        labels = torch.cat((labels, label))

        # PRECISION.append(accuracy(pred.cpu(), label.long()).numpy())

    dcg = dcg_at_k(labels,
                predictions,
                test.user_num,
                items_per_user=items_per_user)

    # acc = np.mean(PRECISION)
    # print(f"acc {acc:.3f}")
    print("Epoch:", epoch, "DCG@2,4,6:", dcg)
    # if epoch > start_checking_epoch and acc > best_acc:

    #     state = {
    #         'net': model.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     torch.save(state,  f"saved_propensity_model/neumf_propensity_{data}.ckpt")
# print(prof.table())