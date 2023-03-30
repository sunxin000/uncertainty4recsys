import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional.classification.accuracy import accuracy
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import dcg_at_k, recall_at_k
from model.neumf import NeuMF
from model.MF import MF
from utils.dataset import ObservedData


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--sample_ratio', type=int, default=1)
    parser.add_argument('--mlp_layers',
                        nargs='*',
                        type=int,
                        default=[64, 32, 16])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--ps_epoch', type=int, default=0)
    parser.add_argument("--n_flag", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dir", default='raw')
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed")
    for k, v in kwargs.items():
        parser.add_argument(k, type=v)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = 1024
    dropout = args.dropout
    n_flag = args.n_flag
    dir = args.dir
    lr = args.lr
    mlp_layers = args.mlp_layers
    embedding_size = args.embedding_size
    data = args.dataset
    sample_ratio = args.sample_ratio
    epoch = args.epoch if data == "coat" else 10
    ps_epoch = args.ps_epoch
    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_ce = 1000000
    items_per_user = 16 if data == "coat" else 10
    dir = args.dir
    propensity = torch.load(
        f'propensity/{dir}/{data}.pt'
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

    model = NeuMF(user_num,
                  item_num,
                  embedding_size,
                  embedding_size,
                  mlp_layers,
                  dropout=0)
    # model = MF(user_num, item_num, embedding_size)
    #! dont knwo whether the testset has unknown user
    #? no
    model = model.to(device)
    loss_func = nn.BCELoss(reduction='none')
    # loss_func = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay)  #! can adjust the weight_decay

    batches = len(train_loader)

    # patient = 20
    # start_checking_epoch = 10
    # suffix = f'_{args.n_flag}' if args.n_flag > 0 else ''

    writer = SummaryWriter(
        log_dir=
        f'tensorboard/val'
    )

    end_epoch = epoch
    for epoch in tqdm(range(1, end_epoch + 1)):
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
        # cur_acc = np.mean(acc)
        loss_tmp /= batches
        # print(f"train acc {cur_acc}")
        # writer.add_scalar('train/acc', cur_acc, epoch - 1)
        # writer.add_scalar('train/loss', loss_tmp, epoch - 1)


        model.eval()
        predictions = torch.empty(0)
        labels = torch.empty(0)
        propensities = torch.empty(0)

        for user, item, label, propensity in val_loader:
            user = user.to(device)
            item = item.to(device)

            pred = model(user, item)
            predictions = torch.cat((predictions, pred.detach().cpu()))
            labels = torch.cat((labels, label))
            propensities = torch.cat((propensities, propensity))
        
        ce = torch.mean(loss_func(predictions, labels.float()))# * torch.reciprocal(propensities))
        writer.add_scalar('val_ce', ce, epoch-1)
        if ce < best_ce:
            predictions = torch.empty(0)
            labels = torch.empty(0)
            for user, item, label in test_loader:
                user = user.to(device)
                item = item.to(device)
                pred = model(user, item)
                predictions = torch.cat((predictions, pred.detach().cpu()))
                labels = torch.cat((labels, label))
            best_dcg = dcg_at_k(labels,
                        predictions,
                        test.user_num,
                        items_per_user=items_per_user)
            best_recall = recall_at_k(labels, predictions, test.user_num,
                                items_per_user)
            best_epoch = epoch
            best_ce = ce
    print(best_dcg)
    print(best_recall)
    print(best_epoch)

if __name__ == "__main__":
    main()

