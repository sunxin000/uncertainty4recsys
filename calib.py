
from sklearn.calibration import calibration_curve, CalibrationDisplay
import torch
from torch.utils.data import random_split, DataLoader
from model.neumf import NeuMF
from utils.dataset import Observe
import matplotlib.pyplot as plt
import numpy as np
from train_propensity import parse_args
from tensorflow_probability.python.stats import expected_calibration_error as ece 


def main():
    args = parse_args()
    embedding_size = args.embedding_size
    mlp_dim = args.mlp_dim
    data = args.dataset
    sample_ratio = args.sample_ratio

    batch_size = 1024
    path = f'saved_propensity_model/{data}/neumf_propensity_{sample_ratio}_{embedding_size}_{mlp_dim}.ckpt'


    train = Observe(data, True, sample_ratio=sample_ratio) #! the sample ratio can change
    user_num = train.user_num
    item_num = train.item_num
    train_loader = DataLoader(dataset=train, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)

    lr = 1e-3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuMF(user_num, item_num,  embedding_size, embedding_size, [mlp_dim, mlp_dim, mlp_dim, mlp_dim])
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])

    predictions= torch.empty(0)
    y_true = torch.empty(0)
    model.eval()
    for user, item, label in train_loader:
        pred = model(user, item)
        predictions = torch.cat((predictions, pred.cpu()))
        y_true = torch.cat((y_true, label))

    y_true = np.asarray(y_true, dtype=np.int32)
    
    predictions = predictions.detach().numpy()

    disp = CalibrationDisplay.from_predictions(y_true, predictions, n_bins=10)
    title = f'{data}_{sample_ratio}_{embedding_size}_{mlp_dim}'
    plt.title(title)
    plt.savefig(f"pic/{data}_{sample_ratio}_{embedding_size}_{mlp_dim}.jpg")

    n_bins = 100
    logits = np.vstack((1-predictions, predictions)).T
    logits =np.asarray(logits, dtype=np.float32)
    error = ece(num_bins=n_bins, logits=logits, labels_true=y_true)
    with open('ece.txt', 'a+') as file:
        file.write(f'ece for {title} is {error.numpy()}')
        file.write('\n')

if __name__ == '__main__':
    main()