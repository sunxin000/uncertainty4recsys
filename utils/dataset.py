import numpy as np
import torch
from torch.utils.data import Dataset

def load_data(dataset, train):
    flag = "train" if train else "test"
    if dataset == "coat":
        matrix = np.loadtxt(f"./data/coat/{flag}.ascii", dtype=int)
        #! like that sparse 
        user, item = np.where(matrix)
        target = matrix[user, item]

        target = target.astype(np.float32)

    elif dataset == "yahoo":
        matrix = np.loadtxt(f"./data/yahoo/{flag}.txt", dtype=int,delimiter=",")
        user = matrix[:,0]
        item = matrix[:,1]
        target = matrix[:, 2].astype(np.float32)
        # train_matrix[:, :-1] -= 1
        # test_matrix[:, :-1] -= 1 
        #! the data i got is from 0
    else:
        raise Exception("Only support coat and ya hoo")
    
    return user, item, target

class ObservedData(Dataset):
    "emplicit"
    def __init__(self, dataset = "coat", train = True, implicit = False, threshold = 3, propensity = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.user, self.item, self.target = load_data(dataset, train)
        self.user_num = np.max(self.user) + 1
        self.item_num = np.max(self.item) + 1
        if implicit:
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
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(target, dtype = torch.float)
            )
        else:
            propensity = self.propensity[index]
            return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(target, dtype = torch.float),
            propensity
        )


class Observe(Dataset):
    def __init__(self, dataset = "coat", train= True, sample_ratio = 4) -> None: #TODO: change the sample rate
        super().__init__()
        user, item, _ = load_data(dataset, train)
        self.user_num = np.max(user) + 1
        self.item_num = np.max(item) + 1

        uniform_matrix = np.array([[x,y] for x in np.arange(self.user_num) for y in np.arange(self.item_num)]) #TODO: a better to calculate the cartesian product torch.cartesian_prod(users, items) 
        missing_matrix = np.array(list(set([tuple(x) for x in uniform_matrix]) - set([x for x in zip(user, item)])))#TODO:   better way

        user_missing = missing_matrix[:, 0]
        item_missing = missing_matrix[:, 1]



        missing_num = missing_matrix.shape[0]

        if sample_ratio == -1:
            self.user = np.append(user, user_missing)
            self.item = np.append(item, item_missing)
            self.target = np.array([1] * len(user) + [0]*len(user_missing)) #! whether observed

        elif sample_ratio == 0:
            self.user = user
            self.item = item
            self.target = np.array([1] * len(user))

        else:
            self.missing_num = min(missing_num, sample_ratio * len(user))
            index = np.random.choice(missing_num, self.missing_num, replace = True)
            self.user = np.append(user, user_missing[index])
            self.item = np.append(item, item_missing[index])
            self.target = np.array([1] * len(user) + [0] * self.missing_num)

    def _neg_sampling(self, sample_ratio): #TODO: reconstruction
        pass

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        target = self.target[index]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(target, dtype = torch.long)
        )










       
        

                

