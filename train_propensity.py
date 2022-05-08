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
from netcal.metrics import ECE
from sklearn.calibration import calibration_curve, CalibrationDisplay
from matplotlib import pyplot as plt


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--sample_ratio', type=int, default=-1)
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
    # weight_decay = args.weight_decay
    writer = SummaryWriter(
        log_dir=
        f'tensorboard/{data}_ps/{embedding_size}_{mlp_layers}_{sample_ratio}_dropout_{args.dropout}')

    epoch = 100 if data == "coat" else 50

    train = Observe(data, True, sample_ratio=sample_ratio)
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

    optimizer = optim.Adam(model.parameters(), lr=lr,)
                        #    weight_decay=weight_decay)

    best_hr = 0
    len_preds = 311704 if data == 'yahoo' else 6960
    batches = len(train_loader)

    ece = ECE(bins=100)

    for epoch in tqdm(range(1, epoch + 1)):
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
            loss = loss_func(prediction, label.float())
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

        if epoch % 10 == 0:
            # display
            torch.save({'net': model.state_dict()}, f"saved_propensity_model/{data}/{dir}/neumf_{sample_ratio}_{epoch}_{n_flag}_with_seed.ckpt")
            if args.visualize:
                disp = CalibrationDisplay.from_predictions(labels,
                                                        preds,
                                                        n_bins=20,
                                                        strategy='quantile')
                title = f'{data}_{sample_ratio}_{embedding_size}_{mlp_layers}'
                plt.title(title)
                plt.savefig(f"pic/coat/small/{sample_ratio}_neumf_{epoch}_20_quantile_dropout{args.dropout}.jpg")
            # generate the propensity
            if args.gen_ps:
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
                    f"data/propensity/{dir}/{data}_epoch_{epoch}_{sample_ratio}_dropout_{args.dropout}.pt")

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()