import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate

import numpy as np
import pickle

from graph.wiki import get_encodings
from rankingmodel.data_preprocess import DataPreprocess
from rankingmodel.models import TemporalSAGE

DP = DataPreprocess()

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kargs):
        device = kargs.pop("device", None)
        kargs["collate_fn"] = lambda x: list(map(lambda x: x.to(device), default_collate(x)))
        super(DataLoader, self).__init__(dataset, **kargs)

def get_dataloader(data, batch_size=16, window_size=30, split_ratio=(0.6,0.2,0.2), device="cpu"):
    x_train, x_val, x_test, y_train, y_val, y_test = DP.get_data(data, window_size)
    # x_train.shape = [#datapoints, window_size, N, n_features]
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(x_train, y_train)

    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    test_dataset = TensorDataset(x_test, y_test)

    x_val = torch.FloatTensor(x_val)
    y_val = torch.FloatTensor(y_val)
    val_dataset = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, device=device, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, device=device, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, device=device, drop_last=True)

    N, n_features = x_train.size()[2:]
    return (N, n_features), (train_dataloader, test_dataloader, val_dataloader)
