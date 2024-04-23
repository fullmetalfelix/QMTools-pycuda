
import time
import torch
from torch.optim import Adam
from torch import nn
from qmtools.automaton_pytorch import AutomatonPT

if __name__ == "__main__":

    n_batch = 1
    nx, ny, nz = (150, 150, 150)
    device = 'cuda'

    model = AutomatonPT(n_layer=4, device=device).half()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.rand(n_batch, nx, ny, nz, 2).to(device).half()

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