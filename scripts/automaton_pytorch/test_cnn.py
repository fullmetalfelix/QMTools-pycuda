
import time
import torch
from torch.optim import Adam
from torch import nn
from qmtools.automaton_pytorch import DensityCNN

if __name__ == "__main__":

    n_batch = 1
    nx, ny, nz = (128, 128, 128)
    device = 'cuda'

    model = DensityCNN().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.rand(n_batch, 2, nx, ny, nz).to(device)

    t0 = time.perf_counter()
    torch.cuda.memory._record_memory_history()
    t = model(x)
    loss = criterion(x, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    print(time.perf_counter() - t0)
    print(t.shape)