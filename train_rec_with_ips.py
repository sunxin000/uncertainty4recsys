import argparse
from multiprocessing import Pool
from unittest import result

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification.accuracy import accuracy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from model.MF import MF
from model.neumf import NeuMF
from utils.dataset import ObservedData
from utils.metrics import dcg_at_k, recall_at_k, calculate_auc


def parse_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coat")
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--sample_ratio", type=int, default=1)
    parser.add_argument("--mlp_layers", nargs="*", type=int, default=[64, 32, 16])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--ps_epoch", type=int, default=0)
    parser.add_argument("--n_flag", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dir", default="raw")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--path")
    parser.add_argument("--seed")
    parser.add_argument("--snips", action="store_true")
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
    epoch = args.epoch if data == "coat" else 4
    # ps_epoch = args.ps_epoch
    weight_decay = args.weight_decay
    # label_smoothing = args.label_smoothing
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_ce = 1000
    items_per_user = 16 if data == "coat" else 10
    dir = args.dir
    # propensity = torch.load(
    #     f'data/propensity/raw/{data}_epoch_{ps_e/data/sunxin/uncertainty4recsys/propensity/label_smoothingpoch}_1_dropout_0.2_label_smoothing_{label_smoothing}.pt'
    # )

    propensity = torch.load(args.path)

    train = ObservedData(data, train='train', implicit=True, propensity=propensity)

    test = ObservedData(data, train='test', implicit=True)
    user_num, item_num = train.user_num, train.item_num

    # train_size = int(0.9 * len(train))
    # validation_size = len(train) - train_size
    # train, validation = random_split(train, [train_size, validation_size])
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    # val_loader = DataLoader(dataset=validation,
    #                         batch_size=batch_size,
    #                         shuffle=True,
    #                         num_workers=0,
    #                         pin_memory=True)
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = NeuMF(
        user_num, item_num, embedding_size, embedding_size, mlp_layers, dropout=dropout
    )
    # model = MF(user_num, item_num, embedding_size)
    #! dont knwo whether the testset has unknown user
    # ? no
    model = model.to(device)
    loss_func = nn.BCELoss(reduction="none")
    # loss_func = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )  #! can adjust the weight_decay

    batches = len(train_loader)

    # patient = 20
    start_checking_epoch = 10
    suffix = f"_{args.n_flag}" if args.n_flag > 0 else ""

    res_suffix = f"_snips" if args.snips else ""
    writer = SummaryWriter(
        log_dir=f"tb_result/{data}_rec_with_ps/{dir}/{data}/NEUMF{suffix}"  # mc_{ps_epoch}_label_smoothing_{label_smoothing}_{n_flag}_new'
    )

    end_epoch = epoch
    check_epoch = 200 if data == "coat" else 1

    snips_denominator = 0
    for *_, propensity in train_loader:
        propensity = propensity.to(device)
        InvP = torch.reciprocal(propensity)
        snips_denominator += torch.sum(InvP)

    # 在训练循环外提前将propensity移到GPU
    propensity = propensity.to(device)
    
    for epoch in tqdm(range(1, end_epoch + 1)):
        model.train()
        loss_tmp = 0
        acc = []
        # 使用prefetch_generator来预加载数据
        for batch_idx, (user, item, label, batch_propensity) in enumerate(train_loader):
            # 批量移动数据到GPU
            user, item, label = user.to(device), item.to(device), label.to(device)
            batch_propensity = batch_propensity.to(device)
            
            # 计算前同步
            torch.cuda.synchronize()
            
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            prediction = model(user, item)
            loss = loss_func(prediction, label.float())
            InvP = torch.reciprocal(batch_propensity)
            loss_ips = torch.sum(loss * InvP)
            if not args.snips:
                snips_denominator = 1
            loss_ips = loss_ips / snips_denominator
            
            # 使用混合精度训练
            loss_ips.backward()
            optimizer.step()
            
            # 计算accuracy时不需要梯度
            with torch.no_grad():
                acc.append(accuracy(prediction, label.long(), task='binary').cpu().numpy())
            
            loss_tmp += loss_ips.item()
            
            # 定期清理缓存
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        cur_acc = np.mean(acc)
        loss_tmp /= batches
        # print(f"train acc {cur_acc}")
        # writer.add_scalar('train/acc', cur_acc, epoch - 1)
        # writer.add_scalar('train/loss', loss_tmp, epoch - 1)
        with torch.no_grad():
            model.eval()
            predictions = torch.empty(0)
            labels = torch.empty(0)

            for user, item, label in test_loader:
                user = user.to(device)
                item = item.to(device)

                pred = model(user, item)
                predictions = torch.cat((predictions, pred.detach().cpu()))
                labels = torch.cat((labels, label))

            # 计算整体AUC
            overall_auc = roc_auc_score(labels.numpy(), predictions.numpy())
            writer.add_scalar('test/overall_auc', overall_auc, epoch - 1)

            # 计算用户级别的平均AUC
            user_auc = calculate_auc(
                labels.numpy(), 
                predictions.numpy(), 
                test.user_num, 
                items_per_user=items_per_user, 
                dataset=data, 
                user_ids=test.user
            )
            writer.add_scalar('test/user_auc', user_auc, epoch - 1)

            dcg = dcg_at_k(
                labels, predictions, test.user_num, items_per_user=items_per_user, dataset=data, user_ids=test.user
            )
            recall = recall_at_k(labels, predictions, test.user_num, items_per_user, dataset=data, user_ids=test.user)
            for k in [2, 4, 6]:
                writer.add_scalar(f"test/dcg_at_{k}", dcg[k // 2 - 1], epoch - 1)
                writer.add_scalar(f"test/recall_at_{k}", recall[k // 2 - 1], epoch - 1)

        if epoch == check_epoch:
            with open(f"new_results/{data}_ips_{dir}{res_suffix}.txt", "a") as f:
                f.write(f"{overall_auc} {user_auc} ")  # 写入两种AUC
                for i in range(3):
                    f.write(str(dcg[i]))
                    f.write(" ")
                for i in range(3):
                    f.write(str(recall[i]))
                    f.write(" ")
                f.write("\n")
    return dcg, recall


if __name__ == "__main__":
    main()
