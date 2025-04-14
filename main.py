import pandas as pd

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *

# import matplotlib
# matplotlib.use('TkAgg')
# # matplotlib.use('Agg')

import matplotlib.pyplot as plt

from mainfunctions import *

L, K = 3, 5
learn_rate = 0.0003
loss_alpha = 0.5
num_of_cpu_workers = 12
cpu, cuda = "cpu", "cuda"
kwargs = {'num_workers': num_of_cpu_workers, 'pin_memory': True}
Generator = torch.Generator(device=cpu)
torch.set_printoptions(edgeitems=15) 
torch.set_default_device(device=cpu)

df = pd.read_csv('lorenz_attractor.csv', index_col=0).to_numpy()
df = torch.from_numpy(df).float().view(-1)
alphas = create_alpha_list(L, K, prefix=True, reverse=True)
alphas = alphas[::10]
print(alphas)
N = alphas.shape[0]

z_dataset = ZTD(df, alphas=alphas, L=L, K=K, forecast_horizon=1, backward_indexation=False)
train_loader = DataLoader(z_dataset, batch_size=1, shuffle=False, generator=Generator, **kwargs)

class NN(nn.Module):
    def __init__(self, N, L, K, loss_alpha = 0.8):
        super().__init__()

        self.la = loss_alpha

        self.N = N
        self.L, self.K = L, K
        self.Linears = nn.ModuleList([nn.Linear(L, 1) for _ in range(self.N)])
        self.Relu = nn.ReLU()
        self.LinearHidden = nn.Linear(self.N, 50)
        self.LinearOutput = nn.Linear(50, 1)

    def forward(self, x):
        modules_res = []
        for i in range(self.N):
            modules_res.append(self.Linears[i](x[i]))
        
        x = torch.cat(modules_res, dim=0)
        x = self.Relu(x)
        x = self.LinearHidden(x)
        x = self.Relu(x)
        x = self.LinearOutput(x)
        return x, modules_res


model = NN(N=N, L=L, K=K, loss_alpha=loss_alpha)
model = model.to(cuda)
criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=learn_rate)

for xb, yb in train_loader:
    print(xb.shape, yb.shape)
    xb = xb.squeeze(0)
    yb = yb.squeeze(0)
    xb = xb.to(cuda) 
    yb = yb.to(cuda)
    xb, modules_xbs = model(xb)
    
    modules_loss = sum(criterion(xb_i, yb) for xb_i in modules_xbs) / len(modules_xbs)
    final_loss = criterion(xb, yb)

    total_loss = model.la * final_loss + (1 - model.la) * modules_loss
    
    print(round(total_loss.item(), 2))

    total_loss.backward()
    optim.step()
    optim.zero_grad()





