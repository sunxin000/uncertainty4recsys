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
from model import NeuMF, MF
from utils.dataset import Observe, ObservedData


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--mlp_layers',
                        nargs='*',
                        type=int,
                        default=[64, 32, 16])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument("--n_flag", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dir", default='raw')
    parser.add_argument("--epoches", type=int, default=0)
    parser.add_argument("--basebone", default='neumf')
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

    epoch = args.epoches 
    if epoch == 0: 
        epoch = 200 if data == "coat" else 100
    weight_decay = args.weight_decay
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    items_per_user = 16 if data == "coat" else 10
    dir = args.dir
    propensity = torch.load(
        f'propensity/{dir}/{data}.pt'
    )
    train_il = ObservedData(data,
                         train=True,
                         implicit=True,
                         propensity=propensity)
    train_pred = Observe(data, train=True, sample_ratio=6, eib=True, propensity=propensity)

    test = ObservedData(data, train=False, implicit=True)
    user_num, item_num = train_il.user_num, train_il.item_num

    train_size = int(0.9 * len(train_il))
    validation_size = len(train_il) - train_size
    train_il, validation = random_split(train_il, [train_size, validation_size])
    il_loader = DataLoader(dataset=train_il,
                              batch_size=batch_size,
                              shuffle=True,
                            #   num_workers=8,
                            #   pin_memory=True
                              )
    

    pred_loader = DataLoader(
        dataset=train_pred,
        batch_size=batch_size,
        shuffle=True,
    #     num_workers=8,
    #    pin_memory=True,
    )

    val_loader = DataLoader(dataset=validation,
                            batch_size=batch_size,
                            shuffle=True,
                            # num_workers=8,
                            # pin_memory=True
                            )
    
    test_loader = DataLoader(dataset=test,
                             batch_size=batch_size,
                             shuffle=False,
                            #  num_workers=8,
                            #  pin_memory=True
                             )

    if basebone == 'neumf':
        model_il = NeuMF(user_num,
                    item_num,
                    embedding_size,
                    embedding_size,
                    mlp_layers,
                    dropout=dropout)
        model_pred = NeuMF(user_num,
                    item_num,
                    embedding_size,
                    embedding_size,
                    mlp_layers,
                    dropout=dropout)
    elif basebone == 'mf':
        model_il = MF(user_num, item_num, embedding_size)
        model_pred = MF(user_num, item_num, embedding_size)
    
    else: raise Exception('only support neumf and mf')

    model_il = model_il.to(device)
    model_pred = model_pred.to(device)

    loss_func = nn.BCELoss(reduction='none')
    optimizer_il = optim.Adam(
        model_il.parameters(), lr=lr,
        weight_decay=weight_decay)  #! can adjust the weight_decay

    optimizer_pred = optim.Adam(
        model_pred.parameters(), lr=lr,
        weight_decay=weight_decay)

    batches = len(pred_loader)

    # patient = 20
    writer = SummaryWriter(
        log_dir=
        f'tb_result/MRDR/{data}_{dir}_{basebone}/{n_flag}'
    )

    best_ce = 1000000
    for epoch in tqdm(range(1, epoch + 1)):
        model_il.train()
        model_pred.train()
        loss_tmp = 0
        acc = []
        for user, item, label, p in il_loader:
            # user = user.to(device)
            # item = item.to(device)
            # label = label.to(device)
            # p = p.to(device)
            model_il.load_state_dict(model_pred.state_dict()) # copy parameter
            optimizer_il.zero_grad()
            pred_il = model_il(user, item)
            cross_entropy_il = loss_func(pred_il, label)
            loss_il = cross_entropy_il / p
            loss_mrdr = cross_entropy_il * (1-p) / p
            loss_il = torch.sum(loss_il)
            loss_il_mrdr = torch.sum(loss_mrdr)
            loss_il_mrdr.backward()
            optimizer_il.step()

        for user, item, o,  label, p in pred_loader:
            # user = user.to(device)
            # item = item.to(device)
            # label = label.to(device)
            # p = p.to(device)
            # o = o.to(device)

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

        model_pred.eval()
        predictions = torch.empty(0, device=device)
        labels = torch.empty(0, device=device)
        propensities = torch.empty(0, device=device)

        for user, item, label, propensity in val_loader:
            pred = model_pred(user, item)
            predictions = torch.cat((predictions, pred))
            labels = torch.cat((labels, label))
            propensities = torch.cat((propensities, propensity))
        
        ce = torch.sum(loss_func(predictions, labels.float())) # * torch.reciprocal(propensities))

        writer.add_scalar('val_ce', ce, epoch-1)

        predictions = torch.empty(0, device=device)
        labels = torch.empty(0, device=device)
        for user, item, label in test_loader:
            # user = user.to(device)
            # item = item.to(device)
            pred = model_pred(user, item)
            predictions = torch.cat((predictions, pred))
            labels = torch.cat((labels, label))
        labels = labels.detach().cpu()
        predictions = predictions.detach().cpu()
        dcg = dcg_at_k(labels,
                    predictions,
                    test.user_num,
                    items_per_user=items_per_user)
        recall = recall_at_k(labels, predictions, test.user_num,
                            items_per_user)

        for k in [2, 4, 6]:
            writer.add_scalar(f'test/dcg_at_{k}', dcg[k // 2 - 1], epoch - 1)
            writer.add_scalar(f'test/recall_at_{k}', recall[k // 2 - 1],
                            epoch - 1)
        
        if ce < best_ce:
            best_dcg = dcg
            best_recall = recall
            best_epoch = epoch
            best_ce = ce
    
    with open(f'results/{data}_MRDR_{dir}_{basebone}_val.txt', 'a') as f:
        for i in range(3):
            f.write(str(best_dcg[i]))
            f.write(' ')
        for i in range(3):
            f.write(str(best_recall[i]))
            f.write(' ')
        f.write('\n')
        

if __name__ == "__main__":
    main()