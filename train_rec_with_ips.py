import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.classification.accuracy import accuracy
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import metrics, dcg_at_k
from model.neumf import NeuMF
from utils.dataset import ObservedData
from train_propensity import parse_args


def main():
    args = parse_args()
    batch_size = 1024
    lr = 1e-3
    n_layers = 4
    embedding_size = args.embedding_size
    data = args.dataset
    mlp_dim = args.mlp_dim
    sample_ratio = args.sample_ratio
    epoch = 1000 if data == "coat" else 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    items_per_user = 16 if data == "coat" else 10

    propensity = torch.load(
        f"data/propensity/neumf_propensity_{sample_ratio}_{data}_{embedding_size}_{mlp_dim}.pt"
    )
    train = ObservedData(data,
                         train=True,
                         implicit=True,
                         propensity=propensity)

    # train = ObservedData(data, train=True, implicit=True)
    test = ObservedData(data, train=False, implicit=True)
    user_num, item_num = train.user_num, train.item_num

    train_size = int(0.9 * len(train))
    validation_size = len(train) - train_size
    train, validation = random_split(train, [train_size, validation_size])
    train_loader = DataLoader(dataset=train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(dataset=validation,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)
    test_loader = DataLoader(dataset=test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    mlp_layers = [mlp_dim] * n_layers
    model = NeuMF(user_num, item_num, embedding_size, embedding_size,
                  mlp_layers)
    #! dont knwo whether the testset has unknown user
    #? no
    model = model.to(device)
    loss_func = nn.BCELoss(reduction='none')
    # loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=0.001)  #! can adjust the weight_decay

    batches = len(train_loader)

    # patient = 20
    start_checking_epoch = 10
    writer = SummaryWriter(
        log_dir=
        f'tensorboard/{data}_rec/{embedding_size}_{mlp_dim}_{sample_ratio}')

    for epoch in tqdm(range(1, epoch + 1)):

        best_acc = 0
        model.train()
        loss_tmp = 0
        acc = []
        for user, item, label, propensity in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_func(prediction, label.float())
            InvP = torch.reciprocal(propensity)
            InvP = InvP.to(device)
            loss_ips = torch.sum(loss * InvP)  #! maybe sum? dont know why
            loss_ips.backward()
            # loss.backward()
            optimizer.step()

            acc.append(accuracy(prediction, label.long()).cpu().numpy())

            loss_tmp += loss_ips.item()
        cur_acc = np.mean(acc)
        loss_tmp /= batches
        # print(f"train acc {cur_acc}")
        writer.add_scalar('train/acc', cur_acc, epoch - 1)
        writer.add_scalar('train/loss', loss_tmp, epoch - 1)

        model.eval()
        PRECISION = []
        predictions = torch.empty(0)
        labels = torch.empty(0)

        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)

            pred = model(user, item)
            predictions = torch.cat((predictions, pred.detach().cpu()))
            labels = torch.cat((labels, label))
            PRECISION.append(accuracy(pred.cpu(), label.long()).numpy())

        dcg = dcg_at_k(labels,
                       predictions,
                       test.user_num,
                       items_per_user=items_per_user)
        for k in [2, 4, 6]:
            writer.add_scalar(f'test/dcg_at_{k}', dcg[k//2-1], epoch-1) 
        acc = np.mean(PRECISION)
        writer.add_scalar('test/acc', acc, epoch-1)
        # print(f"acc {acc:.3f}")
        # print("Epoch:", epoch, "DCG@2,4,6:", dcg)

        # if epoch > start_checking_epoch and acc > best_acc:

        #     state = {
        #         'net': model.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch,
        #     }
        #     torch.save(state,  f"saved_propensity_model/neumf_propensity_{data}.ckpt")


if __name__ == "__main__":
    main()
