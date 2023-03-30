import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.classification.accuracy import accuracy
from tqdm import tqdm

from model import MF, NeuMF
from utils.dataset import Observe, ObservedData

# from torch.utils.tensorboard import SummaryWriter
from utils.metrics import dcg_at_k, recall_at_k


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coat")
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--sample_ratio", type=int, default=1)
    parser.add_argument("--mlp_layers", nargs="*", type=int, default=[64, 32, 16])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ps_epoch", type=int, default=100)
    parser.add_argument("--n_flag", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dir", default="raw")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--basebone", default="neumf")
    parser.add_argument("--path")
    for k, v in kwargs.items():
        parser.add_argument(k, type=v)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.n_flag)
    batch_size = 1024
    dropout = args.dropout
    n_flag = args.n_flag
    dir = args.dir
    lr = args.lr
    basebone = args.basebone
    mlp_layers = args.mlp_layers
    embedding_size = args.embedding_size
    data = args.dataset
    sample_ratio = args.sample_ratio
    epoch = args.epoch if data == "coat" else 6
    ps_epoch = args.ps_epoch
    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    items_per_user = 16 if data == "coat" else 10
    dir = args.dir
    check_epoch = {200} if data == "coat" else {0, 1, 2, 3, 4, 5}
    propensity = torch.load(f"propensity/{dir}/{data}.pt")
    train_il = ObservedData(data, train=True, implicit=True, propensity=propensity)
    train_pred = Observe(
        data, train=True, sample_ratio=6, eib=True, propensity=propensity
    )

    # train = ObservedData(data, train=True, implicit=True)
    test = ObservedData(data, train=False, implicit=True)
    user_num, item_num = train_il.user_num, train_il.item_num

    # train_size = int(0.9 * len(train_il))
    # validation_size = len(train) - train_size
    # train, validation = random_split(train, [train_size, validation_size])
    il_loader = DataLoader(
        dataset=train_il,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    pred_loader = DataLoader(
        dataset=train_pred,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # val_loader = DataLoader(dataset=validation,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         num_workers=0,
    #                         pin_memory=True)

    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if basebone == "neumf":
        model_il = NeuMF(
            user_num,
            item_num,
            embedding_size,
            embedding_size,
            mlp_layers,
            dropout=dropout,
        )
        model_pred = NeuMF(
            user_num,
            item_num,
            embedding_size,
            embedding_size,
            mlp_layers,
            dropout=dropout,
        )
    elif basebone == "mf":
        model_il = MF(user_num, item_num, embedding_size)
        model_pred = MF(user_num, item_num, embedding_size)

    model_il = model_il.to(device)
    model_pred = model_pred.to(device)

    loss_func = nn.BCELoss(reduction="none")
    optimizer_il = optim.Adam(
        model_il.parameters(), lr=lr, weight_decay=weight_decay
    )  #! can adjust the weight_decay

    optimizer_pred = optim.Adam(
        model_pred.parameters(), lr=lr, weight_decay=weight_decay
    )

    batches = len(pred_loader)

    # patient = 20
    best_metric = 0

    for epoch in tqdm(range(1, epoch + 1)):
        model_il.train()
        model_pred.train()
        loss_tmp = 0
        acc = []
        for user, item, label, p in il_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            p = p.to(device)
            model_il.load_state_dict(model_pred.state_dict())  # copy parameter
            optimizer_il.zero_grad()
            pred_il = model_il(user, item)
            cross_entropy_il = loss_func(pred_il, label)
            loss_il = cross_entropy_il / p
            loss_mrdr = cross_entropy_il * (1 - p) / p
            loss_il = torch.sum(loss_il)
            loss_il_mrdr = torch.sum(loss_mrdr)
            loss_il_mrdr.backward()
            optimizer_il.step()

        for user, item, o, label, p in pred_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            p = p.to(device)
            o = o.to(device)

            optimizer_pred.zero_grad()
            label_il = model_il(user, item).detach()
            pred = model_pred(user, item)
            error_il = loss_func(pred, label_il)
            error = loss_func(pred, label)
            dr_loss = error_il + (error - error_il) * o / p
            dr_loss = torch.sum(dr_loss)
            dr_loss.backward()
            optimizer_pred.step()

            acc.append(accuracy(pred, label.long()).cpu().numpy())

            loss_tmp += dr_loss.item()
        cur_acc = np.mean(acc)
        loss_tmp /= batches
        # print(f"train acc {cur_acc}")
        # writer.add_scalar('train/acc', cur_acc, epoch - 1)
        # writer.add_scalar('train/loss', loss_tmp, epoch - 1)
        if epoch in check_epoch:
            model_il.eval()
            model_pred.eval()

            PRECISION = []
            predictions = torch.empty(0)
            labels = torch.empty(0)

            for user, item, label in test_loader:
                user = user.to(device)
                item = item.to(device)

                pred = model_pred(user, item)
                predictions = torch.cat((predictions, pred.detach().cpu()))
                labels = torch.cat((labels, label))
                PRECISION.append(accuracy(pred.cpu(), label.long()).numpy())

            dcg = dcg_at_k(
                labels, predictions, test.user_num, items_per_user=items_per_user
            )
            recall = recall_at_k(labels, predictions, test.user_num, items_per_user)

            if np.mean(dcg) + np.mean(recall) > best_metric:
                best_metric = np.mean(dcg) + np.mean(recall)
                best_dcg = dcg
                best_recall = recall

    with open(f"results/{data}_MRDR_{dir}.txt", "a") as f:
        for i in range(3):
            f.write(str(best_dcg[i]))
            f.write(" ")
        for i in range(3):
            f.write(str(best_recall[i]))
            f.write(" ")
        f.write("\n")


if __name__ == "__main__":
    main()
