from site import check_enableusersite
from sklearn.metrics import precision_recall_curve
from torchmetrics.functional.classification.accuracy import accuracy
from utils.dataset import Observe
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.metrics import metrics
from model.neumf import NeuMF
from netcal.metrics import ECE, MCE
from sklearn.calibration import calibration_curve, CalibrationDisplay
from matplotlib import pyplot as plt


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
    # parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_flag', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
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
    label_smoothing = args.label_smoothing
    # weight_decay = args.weight_decay
    writer = SummaryWriter(
        log_dir=
        f'tensorboard/{data}_ps/{embedding_size}_{mlp_layers}_{sample_ratio}_dropout_{args.dropout}')

    epochs = 100 if data == "coat" else 20

    train = Observe(data, True, sample_ratio=sample_ratio, seed=0)
    # test = Observe(data, False, sample_ratio=sample_ratio)
    train_pos = Observe(data, True, sample_ratio=sample_ratio)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuMF(train.user_num,
                  train.item_num,
                  embedding_size,
                  embedding_size,
                  mlp_layers,
                  dropout=args.dropout)

    # print(len(train))
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
    # test_loader = DataLoader(dataset=test, batch_size=1024, shuffle=True, pin_memory=True)

    #! dont knwo whether the testset has unknown user
    #? no
    model = model.to(device)
    loss_func = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, )
                        #    weight_decay=weight_decay)

    len_preds = len(train_pos.click)
    batches = len(train_loader)
    n_bins = 100
    ece = ECE(n_bins)
    mce = MCE(n_bins)

    check_epochs = 5 if data=='yahoo' else 10
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
            label_float = label.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss = loss_func(prediction, label_float)
            loss.backward()
            optimizer.step()
            acc.append(accuracy(prediction, label).cpu().numpy())
            loss_tmp += loss.item()
            preds.append(prediction.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        labels = np.hstack(labels)
        preds = np.hstack(preds)

        ece_loss = ece.measure(preds, labels)
        writer.add_scalar('train/ece', ece_loss, epoch - 1)

        cur_acc = np.mean(acc)
        writer.add_scalar('loss', loss_tmp / batches, epoch - 1)
        writer.add_scalar('train_acc', cur_acc, epoch - 1)
        # loss_tmp /= batches
        # print(f"Epoch {epoch}: loss {loss_tmp}")
        model.eval()

        _, _, acc = metrics(model, val_loader, 2, device)
        writer.add_scalar('test_acc', acc, epoch - 1)

        if epoch % check_epochs == 0:
            # display
            # torch.save({'net': model.state_dict()}, f"saved_propensity_model/{data}/{dir}/neumf_{sample_ratio}_{epoch}_{n_flag}_with_seed_label_smoothing.ckpt")
            ece_res = ece.measure(preds, labels)
            mce_res = mce.measure(preds, labels)
            with open('thesis/ece.txt', 'a+') as file:
                file.write(f'ece for {epoch} is {ece_res:.4f}')
                file.write('\n')
            with open('thesis/mce.txt', 'a+') as file:
                file.write(f'mce for {epoch} is {mce_res:.4f}')
                file.write('\n')
            if args.visualize:
                disp = CalibrationDisplay.from_predictions(labels,
                                                        preds,
                                                        # strategy='quantile',
                                                        n_bins=20, label='Propensity model')
                                                        # strategy='quantile')
                plt.savefig(f"pic/{data}/{sample_ratio}_neumf_{epoch}_20_dropout{args.dropout}_label_smoothing_{label_smoothing}.jpg")
            # generate the propensity
            if args.gen_ps and epoch==epochs:
                predictions = np.zeros(len_preds)
                with torch.no_grad():
                    for index, (user, item,
                                label) in enumerate(train_loader_not_shuffle):
                        user = user.to(device)
                        item = item.to(device)
                        pred = model(user, item)
                        predictions[index * batch_size:min(
                            (index + 1) * batch_size, len_preds)] = np.asarray(
                                pred.cpu())

                torch.save(
                    predictions,
                    f"data/propensity/{dir}/{data}_epoch_{epoch}_{sample_ratio}_dropout_{args.dropout}_label_smoothing_{label_smoothing}.pt")

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()