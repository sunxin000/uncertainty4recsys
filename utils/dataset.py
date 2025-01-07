import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(dataset, train='train'):
    flag = train
    if dataset == "coat":
        matrix = np.loadtxt(f"./data/coat/{flag}.ascii", dtype=int)
        #! like that sparse
        user, item = np.where(matrix)
        target = matrix[user, item]

        target = target.astype(np.float32)

    elif dataset == "yahoo":
        matrix = np.loadtxt(f"./data/yahoo/{flag}.txt",
                            dtype=int,
                            delimiter=",")
        user = matrix[:, 0]
        item = matrix[:, 1]
        target = matrix[:, 2].astype(np.float32)
        # train_matrix[:, :-1] -= 1
        # test_matrix[:, :-1] -= 1
        #! the data i got is from 0
    elif dataset == "kuairand":
        matrix = np.loadtxt(f"./data/kuairand/{flag}.txt",
                            dtype=int,
                            delimiter=" ")
        user = matrix[:, 0]
        item = matrix[:, 1]
        target = matrix[:, 2].astype(np.float32)
    else:
        raise Exception("Only support coat and yahoo and kuairand")

    return user, item, target


class ObservedData(Dataset):
    "emplicit"

    def __init__(self,
                 dataset="coat",
                 train='train',
                 implicit=False,
                 threshold=3,
                 propensity=None,
                 sample_ratio=0) -> None:
        super().__init__()
        self.threshold = threshold
        self.user, self.item, self.target = load_data(dataset, train)
        self.user_num = np.max(self.user) + 1
        self.item_num = np.max(self.item) + 1
        if implicit and dataset != "kuairand":
            self._preprocess_target()
        self.propensity = propensity

        # self.user_num = np.max(self.user) + 1
        # self.item_num = np.max(self.item) + 1

    def _preprocess_target(self):
        """for implicit recsys"""
        self.target[self.target <= self.threshold] = 0
        self.target[self.target > self.threshold] = 1

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        target = self.target[index]
        if self.propensity is None:
            return (torch.tensor(user, dtype=torch.long),
                    torch.tensor(item, dtype=torch.long),
                    torch.tensor(target, dtype=torch.float))
        else:
            propensity = self.propensity[index]
            return (torch.tensor(user, dtype=torch.long),
                    torch.tensor(item, dtype=torch.long),
                    torch.tensor(target, dtype=torch.float), propensity)


class Observe(Dataset):
    def __init__(self,
                 dataset="coat",
                 train='train',
                 sample_ratio=4,
                 eib=False,
                 propensity=None,
                 seed=None) -> None:  #TODO: change the sample rate
        super().__init__()
        user, item, label = load_data(dataset, train)
        self.user_num = np.max(user) + 1
        self.item_num = np.max(item) + 1
        self.eib = eib
        self.propensity = propensity
        # uniform_matrix = np.array(
        #     [[x, y] for x in np.arange(self.user_num)
        #      for y in np.arange(self.item_num)]
        # )  #TODO: a better to calculate the cartesian product torch.cartesian_prod(users, items)

        observed_set = set(zip(user, item))
        negative_samples = set()
        total_samples = len(user) * sample_ratio
        # 尽量一次性采样足够多的样本，以减少循环次数
        while len(negative_samples) < total_samples:
            # 随机采样用户和物品
            sampled_users = np.random.randint(0, self.user_num, total_samples - len(negative_samples))
            sampled_items = np.random.randint(0, self.item_num, total_samples - len(negative_samples))
            
            # 生成组合并检查是否为负样本
            new_combinations = set(zip(sampled_users, sampled_items)) - observed_set
            
            # 更新负样本集
            negative_samples.update(new_combinations)

        # 由于可能超出所需数量，因此只取所需数量的样本
        negative_samples = list(negative_samples)[:total_samples]
        missing_user = torch.tensor([x[0] for x in negative_samples])
        missing_item = torch.tensor([x[1] for x in negative_samples])
        self.user = torch.cat([torch.tensor(user), missing_user])
        self.item = torch.cat([torch.tensor(item), missing_item])
        self.click  = torch.cat([torch.ones(len(user)), torch.zeros(len(missing_user))])
        # if sample_ratio == -1:
        #     # 使用所有负样本
        #     self.user = torch.cat([user, missing_combinations[:, 0]])
        #     self.item = torch.cat([item, missing_combinations[:, 1]])
        #     self.click = torch.cat([torch.ones(len(user)), torch.zeros(len(missing_combinations))])

        if eib:
            # 如果使用EIB，进行相应处理
            self.target = torch.cat([torch.tensor(label), torch.zeros(total_samples)])
            # 假设这里有一个 _preprocess_target 方法
            if dataset != 'kuairand':
                self._preprocess_target()
            self.propensity = torch.cat([torch.tensor(propensity), torch.ones(total_samples)])
        # user_missing = missing_matrix[:, 0]
        # item_missing = missing_matrix[:, 1]

        # missing_num = missing_matrix.shape[0]

        # if sample_ratio == -1:
        #     self.user = np.append(user, user_missing)
        #     self.item = np.append(item, item_missing)
        #     self.click = np.array(
        #         [1] * len(user) + [0] * len(user_missing))  #! whether observed

        # elif sample_ratio == 0:
        #     self.user = user
        #     self.item = item
        #     self.click = np.array([1] * len(user))

        # else:
        #     self.missing_num = min(missing_num, sample_ratio * len(user))
        #     if seed is not None: np.random.seed(seed)
        #     # np.random.seed(0)
        #     index = np.random.choice(missing_num,
        #                              self.missing_num,
        #                              replace=True)
        #     self.user = np.append(user, user_missing[index])
        #     self.item = np.append(item, item_missing[index])
        #     self.click = np.array([1] * len(user) + [0] * self.missing_num)
        #     if eib:
        #         self.target = np.hstack([label, np.array([0] * self.missing_num)])
        #         self._preprocess_target()
        #         self.propensity = np.hstack([self.propensity, np.array([1.0] * self.missing_num)])

    def _neg_sampling(self, sample_ratio):  #TODO: reconstruction
        pass

    def _preprocess_target(self):
        """for implicit recsys"""
        self.target[self.target <= 3] = 0
        self.target[self.target > 3] = 1

    def __len__(self):
        return len(self.click)


    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        click = self.click[index]
        if self.eib: 
            target = self.target[index]
            propensity = self.propensity[index]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(click, dtype=torch.long),
                torch.tensor(target, dtype=torch.float),
                propensity
            )
        else:
            return (torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(click, dtype=torch.long))

