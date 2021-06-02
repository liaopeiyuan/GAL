from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from helper import ltensor
import pandas as pd
import numpy as np
import os

class GDADataset(Dataset):
    def __init__(self, edge_data, node_data, prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.edge_data = np.ascontiguousarray(edge_data)
        self.node_data = list(node_data)

        data_dir = './data/ml-1m'
        csvs_to_load = ['users.csv', 'movies.csv']
        [users, _] = map(lambda x: pd.read_csv(os.path.join(data_dir, x)),
                              csvs_to_load)

        label_encoder = preprocessing.LabelEncoder()


        self.gender = label_encoder.fit_transform(users.sex.values).reshape(-1, 1)
        self.occupation = users.occupation.values  # transform(users.occupation.values, one_hot_encoder).toarray()
        self.age = users.age.values  # transform(users.age.values, one_hot_encoder).toarray()
        self.age_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

        stack = []

        for user in self.node_data:
            stack.append([user, self.gender[user, :], self.occupation[user], self.age_dict[self.age[user]]])

        users = []
        genders = []
        occupations = []
        ages = []

        for [user, gender, occupation, age] in stack:
            users.append(user)
            genders.append(gender)
            occupations.append(occupation)
            ages.append(age)

        users = ltensor(users)
        genders = ltensor(genders)
        occupations = ltensor(occupations)
        ages = ltensor(ages)

        self.user_features = (users, genders, occupations, ages)

    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, idx):
        return self.edge_data[idx]


class NodeClassification(Dataset):
    def __init__(self, data_split, prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(data_split)

        data_dir = './data/ml-1m'
        csvs_to_load = ['users.csv', 'movies.csv']
        [users, _] = map(lambda x: pd.read_csv(os.path.join(data_dir, x)),
                                                           csvs_to_load)

        label_encoder = preprocessing.LabelEncoder()

        self.gender = label_encoder.fit_transform(users.sex.values).reshape(-1, 1)
        self.occupation = users.occupation.values #transform(users.occupation.values, one_hot_encoder).toarray()
        self.age = users.age.values #transform(users.age.values, one_hot_encoder).toarray()
        self.age_dict = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user = self.dataset[idx]
       
        return [user, self.gender[user,:], self.occupation[user], self.age_dict[self.age[user]]]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()


class KBDataset(Dataset):
    def __init__(self, data_split, prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        np.random.shuffle(data)
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()