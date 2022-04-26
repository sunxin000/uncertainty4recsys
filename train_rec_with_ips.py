import numpy as np
from tqdm import tqdm
import argparse 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.classification.accuracy import accuracy
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import metrics, dcg_at_k
from model.neumf import NeuMF
from utils.dataset import ObservedData
def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--sample_ratio', type=int, default=1)
    parser.add_argument('--mlp_layers',nargs='*' ,type=int, default=[64, 32, 16])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--ps_epoch', type=int, default=50)
    for k, v in kwargs.items():
        parser.add_argument(k, type=v) 
    return parser.parse_args()

def main():
    args = parse_args()
    batch_size = 1024
    lr = args.lr
    mlp_layers = args.mlp_layers
    embedding_size = args.embedding_size
    data = args.dataset
    sample_ratio = args.sample_ratio
    epoch = 100 if data == "coat" else 20
    ps_epoch = args.ps_epoch
    weight_decay = args.weight_decay

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    items_per_user = 16 if data == "coat" else 10
    dir = 'raw'
    propensity = torch.load(
        f'data/propensity/raw/coat_epoch_{ps_epoch}_-1.pt'
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

    model = NeuMF(user_num, item_num, embedding_size, embedding_size,
                  mlp_layers)
    #! dont knwo whether the testset has unknown user
    #? no
    model = model.to(device)
    loss_func = nn.BCELoss(reduction='none')
    # loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)  #! can adjust the weight_decay

    batches = len(train_loader)

    # patient = 20
    start_checking_epoch = 10
    writer = SummaryWriter(
        log_dir=
        f'tensorboard/{data}_rec_with_ps/{sample_ratio}_{embedding_size}_{mlp_layers}_{ps_epoch}_weight_decay_{weight_decay}')

    for epoch in tqdm(range(1, epoch + 1)):
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
