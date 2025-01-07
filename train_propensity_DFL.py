import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification.accuracy import accuracy
from tqdm import tqdm

from model.neumf import NeuMF
from utils.dataset import Observe
from utils.metrics import metrics
from dual_focal_loss import DualFocalLoss


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--sample_ratio', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--gen_ps', action='store_true')
    parser.add_argument('--mlp_layers',
                        nargs='*',
                        type=int,
                        default=[64, 32, 16])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dir', default='raw')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_flag', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=2.0)  # 新增gamma参数
    for k, v in kwargs.items():
        parser.add_argument(k, type=v)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    sample_ratio = args.sample_ratio
    batch_size = 1024
    embedding_size = args.embedding_size
    data = args.dataset
    mlp_layers = args.mlp_layers
    lr = args.lr
    dir = args.dir
    n_flag = args.n_flag
    gamma = args.gamma

    writer = SummaryWriter(
        log_dir=
        f'tensorboard/{data}_ps_DFL/{embedding_size}_{mlp_layers}_{sample_ratio}_dropout_{args.dropout}_gamma_{gamma}'
    )

    epochs = 100 if data == "coat" else 20

    train = Observe(data, 'train', sample_ratio=sample_ratio, seed=0)
    train_pos = Observe(data, 'train', sample_ratio=sample_ratio)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuMF(train.user_num,
                  train.item_num,
                  embedding_size,
                  embedding_size,
                  mlp_layers,
                  dropout=args.dropout)

    train_size = int(0.9 * len(train))
    validation_size = len(train) - train_size
    train, validation = random_split(train, [train_size, validation_size])

    train_loader = DataLoader(dataset=train,
                            batch_size=1024,
                            shuffle=True,
                            pin_memory=True)
    train_loader_not_shuffle = DataLoader(dataset=train_pos,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=True)
    val_loader = DataLoader(dataset=validation,
                          batch_size=1024,
                          shuffle=True,
                          pin_memory=True)

    model = model.to(device)
    loss_func = DualFocalLoss(gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    len_preds = len(train_pos.click)
    batches = len(train_loader)
    n_bins = 100

    check_epochs = 20 if data=='yahoo' or data=='kuairand' else 100
    for epoch in tqdm(range(1, epochs + 1)):
        best_acc = 0
        model.train()
        loss_tmp = 0
        acc = []
        preds = []
        labels = []
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_func(prediction, label)
            loss.backward()
            optimizer.step()
            acc.append(accuracy(prediction, label, task='binary').cpu().numpy())
            loss_tmp += loss.item()
            preds.append(prediction.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        
        labels = np.hstack(labels)
        preds = np.hstack(preds)

        cur_acc = np.mean(acc)
        writer.add_scalar('loss', loss_tmp / batches, epoch - 1)
        writer.add_scalar('train_acc', cur_acc, epoch - 1)

        model.eval()
        _, _, acc = metrics(model, val_loader, 2, device)
        writer.add_scalar('test_acc', acc, epoch - 1)

        if epoch % check_epochs == 0:
            torch.save(
                {'net': model.state_dict()}, 
                f"propensity/saved_model/{data}_{dir}_neumf_DFL_{sample_ratio}_{epoch}_{n_flag}_gamma_{gamma}.ckpt"
            )

            if args.visualize:
                disp = CalibrationDisplay.from_predictions(
                    labels,
                    preds,
                    n_bins=20,
                    label='Propensity model DFL'
                )
                plt.savefig(
                    f"pic/{data}/{sample_ratio}_neumf_DFL_{epoch}_20_dropout{args.dropout}_gamma_{gamma}.jpg"
                )

            if args.gen_ps and epoch==epochs:
                predictions = np.zeros(len_preds)
                with torch.no_grad():
                    for index, (user, item, label) in enumerate(train_loader_not_shuffle):
                        user = user.to(device)
                        item = item.to(device)
                        pred = model(user, item)
                        predictions[index * batch_size:min(
                            (index + 1) * batch_size, len_preds)] = np.asarray(
                                pred.cpu())

                torch.save(
                    predictions,
                    f"propensity/{dir}/{data}_DFL_gamma_{gamma}.pt")

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main() 