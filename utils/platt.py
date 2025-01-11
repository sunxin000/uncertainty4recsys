import numpy as np
import torch
from netcal.metrics import ECE, MCE
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os


class ModelWithPlatt(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 
        self.a = nn.Parameter(torch.FloatTensor([0.6],).cuda())
        self.b = nn.Parameter(torch.FloatTensor([0.]).cuda())


    def forward(self, input):
        logits = self.model(input)
        return self.platt_scale(logits)
    
    def platt_scale(self, logits):
        return torch.mul(logits, self.a) + self.b
    

    def set_ab(self, valid_loader):
        # 创建保存图片的目录
        save_dir = 'platt_calibration_plots'
        os.makedirs(save_dir, exist_ok=True)
        
        nll_criterion = nn.BCEWithLogitsLoss()
        ece_criterion = ECE(bins=100).measure
        
        # 添加记录列表
        nll_history = []
        ece_history = []
        checkpoint_logits = {}
        iterations = []
        
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for user, item, labels in valid_loader:
                user = user.cuda()
                item = item.cuda()
                logits = self.model(user, item)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

            before_platt_nll = nll_criterion(logits, labels.float()).item()
            before_platt_ece = ece_criterion(np.asarray(torch.sigmoid(logits).cpu()), np.asarray(labels.cpu()))

            print(before_platt_nll)
            print(before_platt_ece)

            optimizer = optim.LBFGS([self.a, self.b], lr=0.01, max_iter=1000)

            def eval():
                optimizer.zero_grad()
                positive_logits = logits
                positive_labels = labels
                
                # 计算当前loss
                loss = nll_criterion(self.platt_scale(positive_logits), positive_labels.float())
                
                # 记录当前指标
                current_nll = loss.item()
                current_ece = ece_criterion(
                    np.asarray(torch.sigmoid(self.platt_scale(positive_logits)).detach().cpu()), 
                    np.asarray(positive_labels.cpu())
                )
                
                # 保存历史记录
                nll_history.append(current_nll)
                ece_history.append(current_ece)
                iterations.append(len(nll_history))
                
                # 每10次迭代保存一次logits
                if len(nll_history) % 10 == 0:
                    checkpoint_logits[len(nll_history)] = {
                        'logits': torch.sigmoid(self.platt_scale(positive_logits)).detach().cpu(),
                        'a': self.a.item(),
                        'b': self.b.item(),
                        'nll': current_nll,
                        'ece': current_ece
                    }
                    # 直接保存    torch.save(torch.sigmoid(scaled_logits).detach().cpu(), f'propensity/LS+Platt_Scaling/{data}.pt')
                    torch.save(checkpoint_logits, f'platt_calibration_plots/checkpoint_logits_{len(nll_history)}_no_ls.pt')
                
                loss.backward()
                return loss
                
            optimizer.step(eval)
            
            # 绘制并保存训练曲线
            plt.figure(figsize=(12, 5))
            
            # 创建两个子图
            plt.subplot(1, 2, 1)
            plt.plot(iterations, nll_history, 'b-', label='NLL Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Negative Log Likelihood')
            plt.title('NLL Loss Curve')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(iterations, ece_history, 'r-', label='ECE')
            plt.xlabel('Iterations')
            plt.ylabel('Expected Calibration Error')
            plt.title('ECE Curve')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'calibration_curves_no_ls.png'))
            plt.close()
            
            # 保存指标数据到文本文件
            with open(os.path.join(save_dir, 'calibration_metrics_no_ls.txt'), 'w') as f:
                f.write(f"Final Parameters:\n")
                f.write(f"a = {self.a.item():.6f}\n")
                f.write(f"b = {self.b.item():.6f}\n\n")
                f.write(f"Training History:\n")
                f.write("Iteration\tNLL\tECE\n")
                for i, (nll, ece) in enumerate(zip(nll_history, ece_history)):
                    f.write(f"{i+1}\t{nll:.6f}\t{ece:.6f}\n")
            
            
            after_platt_nll = nll_criterion(self.platt_scale(logits), labels.float()).item()
            after_platt_ece = ece_criterion(np.asarray(torch.sigmoid(self.platt_scale(logits)).cpu()), np.asarray(labels.cpu()))
            print(after_platt_nll)
            print(after_platt_ece)
            print(self.a.item(), self.b.item())
            return self, logits, labels





        
        