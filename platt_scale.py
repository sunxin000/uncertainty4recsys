from ast import parse
import torch
import argparse
from utils.dataset import Observe
from model.neumf import NeuMF
from utils.platt import ModelWithPlatt
from torch.utils.data import random_split, DataLoader 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coat')
    args = parser.parse_args()
    return args
def main():
    # torch.manual_seed(42)
    args = parse_args()
    data = args.dataset
    sample_ratio = 1
    embedding_size = 64

    ckpt = torch.load(f'propensity/saved_model/logits_model/logits_{data}_ls.ckpt')
    train = Observe(data, True, sample_ratio=sample_ratio)

    user_num = train.user_num 
    item_num = train.item_num
    # train_size = int(0.9 * len(train))
    # val_size = len(train) - train_size

    # train, val = random_split(train, [train_size, val_size])


    train_loader = DataLoader(dataset=train, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(dataset=val, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)


    model = NeuMF(user_num, item_num, embedding_size, embedding_size, [64, 32, 16], dropout=0.2, output_logits=True)
    model.load_state_dict(ckpt['net'])
    model = model.cuda()
    scaled_model = ModelWithPlatt(model)
    _, logits, labels = scaled_model.set_ab(train_loader)

    scaled_logits = scaled_model.platt_scale(logits)

    torch.save(torch.sigmoid(logits).cpu(), f'propensity/ls+platt/raw_{data}.pt')
    torch.save(torch.sigmoid(scaled_logits).detach().cpu(), f'propensity/ls+platt/{data}.pt')

    

    print(scaled_model.a, scaled_model.b)

if __name__ == '__main__':
    main()