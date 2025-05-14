from data_load import *
import torch
import torch.nn as nn
from tqdm import tqdm 

L= 3
K= 5
num_of_cpu_workers=   12
hidden_neurons=       50
learn_rate=           1e-3
batch_size=           1
shuffle=              False
custom_loss=          False
custom_loss_rate=     0.5
# ZTD parameters 
backward_index=       False


class NN(nn.Module):
    def __init__(self, L, K, gen, hidden_neurons, fully_connected):
        super().__init__()
        N = K ** L

        
        
        self.FC = nn.Sequential(
            nn.ReLU(),
            nn.Linear(K ** L, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )

    def forward(self, x):
        

        x = self.FC(x)

        return x, x


gen = torch.Generator(device="cpu")
kwargs = {'num_workers': num_of_cpu_workers, 'pin_memory': True}
alphas = create_alpha_list(L, K)
data = pd.read_csv('data/lorenz_attractor.csv', index_col=0).to_numpy()
data = (data - data.min()) / (data.max() - data.min())
data_tensor = torch.from_numpy(data).float().view(-1)
train = ZTD(data_tensor, alphas, backward_index)
train_loader = DataLoader(train, batch_size, shuffle, generator=gen, **kwargs)

model = NN(L, K, gen, hidden_neurons, False)
model = model.to("cuda")

criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=learn_rate)
# scaler = torch.amp.GradScaler(
#     "cuda",
#     enabled=torch.cuda.is_available(),
#     init_scale=2.**16,                    # starting loss scale
#     growth_factor=2.0,                    # multiply scale when no inf/NaN for a while
#     backoff_factor=0.5,                   # divide scale on inf/NaN
#     growth_interval=2000                  # number of steps to wait before growing
# )


for xb, yb in tqdm(train_loader, desc="Training", unit="batch"):
    xb, yb = xb.to("cuda"), yb.to("cuda")
    # with torch.amp.autocast("cuda", dtype=torch.float16):
    xb, modules_xbs = model(xb)

    final_loss = criterion(xb, yb)
    if custom_loss:
        modules_loss = torch.sum([criterion(xb_i, yb) for xb_i in modules_xbs]) / len(modules_xbs)
        train_loss = model.la * final_loss + (1 - custom_loss_rate) * modules_loss  
    else: 
        train_loss = final_loss

    # scaler.scale(train_loss).backward()
    # scaler.step(optim)
    # scaler.update()
    # optim.zero_grad() # check reasonability of the flag set_to_none=True
    train_loss.backward()
    optim.step()
    optim.zero_grad()
    

