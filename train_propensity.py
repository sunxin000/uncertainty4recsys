from torchmetrics.functional.classification.accuracy import accuracy
from utils.dataset import Observe
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from utils.metrics import metrics
from model.neumf import NeuMF


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--sample_ratio', type=int, default=4)
    parser.add_argument('--mlp_dim', type=int, default=32)
    return parser.parse_args()
def main():  
    args = arg_parse()
    sample_ratio = args.sample_ratio
    batch_size = 1024
    embedding_size = args.embedding_size
    mlp_dim = args.mlp_dim
    data = args.dataset
    writer = SummaryWriter(log_dir=f'tensorboard/{data}/{embedding_size}_{mlp_dim}_{sample_ratio}')

    epoch = 500 if data == "coat" else 50

    train = Observe(data, True, sample_ratio=sample_ratio)
    test = Observe(data, False, sample_ratio=sample_ratio)
    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuMF(train.user_num, train.item_num, embedding_size, embedding_size, [mlp_dim, mlp_dim, mlp_dim, mlp_dim])

    print(len(train))
    train_size  = int(0.9 * len(train))
    validation_size = len(train) - train_size
    train, validation = random_split(train, [train_size, validation_size])

    train_loader = DataLoader(dataset=train, batch_size=1024, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=validation, batch_size=1024, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=1024, shuffle=True, pin_memory=True)

    #! dont knwo whether the testset has unknown user
    #? no 
    model = model.to(device)
    loss_func = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= 0.001)

    best_hr = 0
    batches = len(train_loader)

    # patient = 20
    if data == 'coat': 
        start_checking_epoch = 100
    else:
        start_checking_epoch = 10

    for epoch in range(1, epoch + 1):

        best_acc = 0
        model.train()
        loss_tmp = 0
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


            loss_tmp += loss.item()
        cur_acc = np.mean(acc)
        writer.add_scalar('loss', loss_tmp/batches, epoch-1)
        writer.add_scalar('train_acc', cur_acc, epoch-1)
        # loss_tmp /= batches
        # print(f"Epoch {epoch}: loss {loss_tmp}")
        model.eval()

        _, _, acc = metrics(model, val_loader, 2, device)
        writer.add_scalar('test_acc', acc, epoch-1)

        if epoch > start_checking_epoch and acc > best_acc:
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state,  f"saved_propensity_model/neumf_propensity_{sample_ratio}_{data}_{embedding_size}_{mlp_dim}.ckpt")
        print(f"acc {acc:.3f}")


    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()