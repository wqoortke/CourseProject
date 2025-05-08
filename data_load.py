import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import *

# in particular [1, 2, 1] and [6, 2, 1] are just the same observations but shifted, 
# because indeces are i+1, i+3, i+4 and i+6, i+8, i+9 which is all previous but +5
# later I'll try to resolve the problem by fixing creation of alpha so that
# alpha-spaces can not be isomorphic
# other approach is to decrease density of alphas in list,
# so that we extend horizons and keep memory not exceeding limit

def dump_tensor_to_csv(t, path, sep = ",",width = 8,precision = 6):
    arr = t.detach().cpu().numpy()
    with open(path, 'w') as f:
        for row in arr:
            parts = []
            for v in row:
                p = precision - 1 if v < 0 else precision
                if sep == ",":
                    parts.append(f"{v:.{p}f}")
                else:
                    parts.append(f"{v: {width}.{p}f}")
            f.write(sep.join(parts) + "\n")

def generate_alpha(L, K):
    for combination in itertools.product(range(1, K + 1), repeat=L):
        yield list(combination)

def create_alpha_list(L, K, prefix=True, reverse=True):
    alphas = torch.tensor(list(generate_alpha(L, K)), dtype=torch.int32)
    if prefix:
        for i, nums in enumerate(alphas):
            alphas[i] = torch.cumsum(nums, dim=0)
            if reverse:
                alphas[i] = torch.flip(alphas[i], dims=[0])
    return alphas

class ZTD(Dataset):
    """ 
    ZTD - Z-blocks TimeSeries Dataset
    return concatenated Z-vectors int the form of a long single vector
    """
    def __init__(self, data, alphas, backward_indexation=False, forecast_horizon=1):
        self.data = data
        self.alphas = alphas

        self.maxind = torch.max(alphas)
        self.length_of_timeseries = len(data)
        
        self.backward_indexation = backward_indexation
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return self.length_of_timeseries - self.maxind - self.forecast_horizon + 1

    def __getitem__(self, idx):
        if idx >= len(self) or idx < -len(self):
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
            x = torch.cat((x, self.data[idx - alpha]), dim=0)
        y = self.data[idx : idx + self.forecast_horizon]
        
        return x, y

