import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# in particular [1, 2, 1] and [6, 2, 1] are just the same observations but shifted, 
# because indeces are i+1, i+3, i+4 and i+6, i+8, i+9 which is all previous but +5
# later I'll try to resolve the problem by fixing creation of alpha so that
# alpha-spaces can not be isomorphic
# other approach is to decrease density of alphas in list,
# so that we extend horizons and keep memory not exceeding limit

def generate_alpha(L, K):
    for combination in itertools.product(range(1, K + 1), repeat=L):
        yield list(combination)

def create_alpha_list(L, K, prefix=False, reverse=False):
    alphas = torch.tensor(list(generate_alpha(L, K)), dtype=torch.int32)
    if prefix:
        for i, nums in enumerate(alphas):
            alphas[i] = torch.cumsum(nums, dim=0)
            if reverse:
                alphas[i] = torch.flip(alphas[i], dims=[0])
    return alphas

class ZTD(Dataset):
    """ZTD - Z-vectors TimeSeries Dataset"""
    def __init__(self, data, alphas, L, K, forecast_horizon=1, backward_indexation=True):
        self.L = L
        self.K = K
        self.data = data
        self.alphas = alphas

        self.maxind = torch.max(alphas)
        self.length_of_timeseries = len(data)
        
        self.backward_indexation = backward_indexation
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return self.length_of_timeseries - self.maxind - self.forecast_horizon + 1

    def __getitem__(self, idx):
        if idx >= len(self) or idx <= -len(self):
            raise IndexError("Index out of range")
        if self.backward_indexation:
            if idx < 0:
                idx = len(self) + idx
            idx = len(self) + self.maxind - idx - 1
        else:
            if idx < 0:
                idx = len(self) + idx
            idx += self.maxind 

        x = torch.tensor([], dtype=torch.float32)

        for alpha in self.alphas:
            x = torch.cat((x, self.data[idx - alpha].unsqueeze(0)), dim=0)
        y = self.data[idx : idx + self.forecast_horizon]
        
        return x, y


