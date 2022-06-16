import argparse
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchmetrics.functional.classification.accuracy import accuracy

from torch.utils.tensorboard import SummaryWriter
from model.neumf import NeuMF

from utils.metrics import metrics, dcg_at_k, recall_at_k
from utils.dataset import ObservedData

torch.backends.cudnn.benchmark = True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dataset", default='yahoo')
    parser.add_argument("--n_flag", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=200)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    weight_decay = args.weight_decay
    lr = args.lr
    batch_size = args.batch_size
    data = args.dataset
    dropout = args.dropout
    epoch = args.epoch if data == "coat" else 10
    items_per_user = 16 if data == "coat" else 10
    # n_layers = 4
    # mlp_dim = 64 if data == 'coat' else 128
    # mlp_layers = [mlp_dim] * n_layers
    # embedding_size = 128 if data == 'coat' else 64
    mlp_layers = [64, 32, 16]
    embedding_size = 64

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

    model = NeuMF(user_num, item_num, embedding_size, embedding_size,
                  mlp_layers, dropout=dropout)
    # model = MF(user_num, item_num, embedding_size)
    model = model.cuda()
    #! dont knwo whether the testset has unknown user
    #? no
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank])

    # loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  #! can adjust the weight_decay
    loss_func = nn.BCELoss()
    batches = len(train_loader)
    # patient = 20
    suffix = f'_{args.n_flag}' if args.n_flag > 0 else ''
    writer = SummaryWriter(
        log_dir=f'tensorboard/{data}_benchmark_recall/weight_decay_{weight_decay}_lr_{lr}_dropout_{dropout}_{suffix}')

    for epoch in tqdm.tqdm(range(1, epoch + 1)):
        # with profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        model.train()
        loss_tmp = 0
        acc = []
        for user, item, label in train_loader:
            user, item = user.cuda(), item.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_func(prediction, label.float())
            loss.backward()
            optimizer.step()
            acc.append(accuracy(prediction, label.long()).cpu().numpy())
            loss_tmp += loss.item()
        loss_tmp /= batches
        cur_acc = np.mean(acc)
        writer.add_scalar('train/acc', cur_acc, epoch - 1)
        writer.add_scalar('train/loss', loss_tmp, epoch - 1)

        model.eval()

        PRECISION = []
        predictions = torch.empty(0)
        labels = torch.empty(0)
        for user, item, label in test_loader:
            user, item = user.cuda(), item.cuda()
            pred = model(user, item)
            predictions = torch.cat((predictions, pred.detach().cpu()))
            labels = torch.cat((labels, label))
            PRECISION.append(accuracy(pred.cpu(), label.long()).numpy())

        dcg = dcg_at_k(labels,
                       predictions,
                       test.user_num,
                       items_per_user=items_per_user)
        recall = recall_at_k(
            labels,
            predictions,
            test.user_num,
            items_per_user)
        for k in [2, 4, 6]:
            writer.add_scalar(f'test/dcg_at_{k}', dcg[k//2-1], epoch-1) 
            writer.add_scalar(f'test/recall_at_{k}', recall[k//2-1], epoch-1)
        acc = np.mean(PRECISION)
        writer.add_scalar('test/acc', acc, epoch-1)
        # if epoch > start_checking_epoch and acc > best_acc:

        #     state = {
        #         'net': model.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch,
        #     }
        #     torch.save(state,  f"saved_propensity_model/neumf_propensity_{data}.ckpt")
    # print(prof.table())
if __name__ == "__main__":
    main()