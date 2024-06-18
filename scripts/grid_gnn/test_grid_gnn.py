import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.append("../..")

from qmtools.pt.gnn import MPNNEncoder
from qmtools.pt.grid import AtomGrid, DensityGridNN, GridLoss, lorentzian, lorentzian2

if __name__ == "__main__":

    device = "cuda"

    mpnn_encoder = MPNNEncoder(device=device, node_embed_size=128, n_class=7)
    model = DensityGridNN(
        mpnn_encoder,
        proj_channels=[64, 32, 2],
        cnn_channels=[64, 32, 8],
        lorentz_type=2,
        per_channel_scale=True,
        device=device,
    )
    criterion = GridLoss(grad_factor=1)
    # model = torch.compile(model)

    n_batch = 2
    pos = torch.rand(15, 3, device=device)
    classes = torch.rand(15, 7, device=device)
    edges = torch.tensor(
        [
            [0, 1, 4, 6, 2, 4, 5, 2],
            [5, 14, 8, 2, 5, 3, 4, 12],
        ],
        device=device,
    )
    origin = torch.zeros(n_batch, 3)
    lattice = torch.stack([torch.eye(3, device=device) for _ in range(n_batch)], dim=0)
    atom_grid = AtomGrid(pos, origin, lattice, grid_shape=(120, 120, 120), device=device)
    batch_nodes = [5, 10]

    # Test memory usage
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.memory._record_memory_history()

    with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
        print(classes.dtype, edges.dtype)
        x = model(atom_grid, classes, edges, batch_nodes)
        loss = criterion(x, torch.rand_like(x))

    loss[0].backward()

    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")  # Visualize using https://pytorch.org/memory_viz
    print(time.perf_counter() - t0)

    # Profile execution time
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        x = model(atom_grid, classes, edges, batch_nodes)
        loss = criterion(x, torch.rand_like(x))
        loss[0].backward()
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Check correctness of lorentzian gradient calculation
    distance = torch.rand(1, 100, 15, 1, requires_grad=True, dtype=torch.float64)
    scale = torch.rand(1, 1, 1, 10, requires_grad=True, dtype=torch.float64)
    amplitude = torch.rand(1, 1, 1, 10, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(lorentzian, (distance, scale))
    torch.autograd.gradgradcheck(lorentzian, (distance, scale))
    torch.autograd.gradcheck(lorentzian2, (distance, scale, amplitude))
    torch.autograd.gradgradcheck(lorentzian2, (distance, scale, amplitude))

    l = lorentzian(distance, scale)
    assert l.shape == (1, 100, 15, 10)

    l = lorentzian2(distance, scale, amplitude)
    assert l.shape == (1, 100, 15, 10)
