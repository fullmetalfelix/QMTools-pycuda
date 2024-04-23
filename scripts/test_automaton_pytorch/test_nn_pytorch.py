
import time
import torch
from qmtools.automaton_pytorch import AutomatonPT

if __name__ == "__main__":

    n_batch = 1
    nx, ny, nz = (50, 50, 50)
    device = 'cuda'

    model = AutomatonPT(n_layer=4, device=device)
    x = torch.rand(n_batch, nx, ny, nz, 2).to(device)

    t0 = time.perf_counter()
    t = model(x)
    print(time.perf_counter() - t0)
    print(t.shape)