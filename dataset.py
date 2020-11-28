import torch
from torch.utils.data import Dataset
import joblib

# redefine Dataset of Torch for data loading
class MyDataset(Dataset):
    def __init__(self, file_name, keys=None, transform=False):
        # load preprocessing data
        with open(file_name, 'rb') as f:
            self.data = joblib.load(f)
        self.keys = keys
        self.log_mel = self.data[self.keys[0]]
        # self.target = self.data[self.keys[1]]
        self.transform = transform

    def __getitem__(self, item):
        inputs, target = self.log_mel[item], self.log_mel[item]
        return inputs, target

    def __len__(self):
        return len(self.log_mel)


if __name__ == '__main__':
    file_name = './data/pre_data/pump.db'
    dataset = MyDataset(file_name, keys=['log_mel'])
