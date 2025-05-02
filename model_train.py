from data_load import *
import torch
import torch.nn as nn
from tqdm import tqdm 
from box import Box

globals().update(Box.from_yaml(filename="params.yaml"))


class NN(nn.Module):
    def __init__(self, L, K, la, hidden_neurons):
        super().__init__()
        self.L, self.K = L, K
        self.la = la
        self.ZBW = nn.Parameter(torch.randn(L, K ** L) * 0.01) # Z_block_weights
        self.ZBB = nn.Parameter(torch.randn(1, L) * 0.01)      # Z_block_bias
        self.FC = nn.Sequential(
            nn.ReLU(),
            nn.Linear(K ** L, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )

    def forward(self, x):
        print(x.shape)
        x_m = self.ZBW @ x 
        print(x_m.shape)
        x_m = x_m + self.ZBB
        print(x_m.shape)
        x = self.FC(x_m)
        return x, x_m

cpu, cuda = "cpu", "cuda"
Generator = torch.Generator(device=cpu)
kwargs = {'num_workers': num_of_cpu_workers, 'pin_memory': True}

alphas = create_alpha_list(L, K)
data = pd.read_csv('data/lorenz_attractor.csv', index_col=0).to_numpy()
data_tensor = torch.from_numpy(data).float().view(-1)
train = ZTD(data_tensor, alphas, L, K)
train_loader = DataLoader(train, batch_size, shuffle, generator=Generator, **kwargs)

model = NN(L, K, la, hidden_neurons)
model = model.to(cuda)

criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=learn_rate)

for xb, yb in tqdm(train_loader, desc="Training", unit="batch"):
    xb, yb = xb.squeeze(0), yb.squeeze(0)
    xb, yb = xb.to(cuda), yb.to(cuda)
    xb, modules_xbs = model(xb)
    modules_loss = sum(criterion(xb_i, yb) for xb_i in modules_xbs) / len(modules_xbs)
    final_loss = criterion(xb, yb)

    if custom_loss:
        train_loss = model.la * final_loss + (1 - model.la) * modules_loss  
    else: 
        train_loss = final_loss

    train_loss.backward()
    optim.step()
    optim.zero_grad()

