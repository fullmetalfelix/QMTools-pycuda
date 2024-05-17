import time

import torch

import sys
sys.path.append("../..")

from qmtools.pt.gnn import MPNNEncoder
from qmtools.pt.grid import AtomGrid, DensityGridNN

if __name__ == "__main__":

    device = "cuda"

    mpnn_encoder = MPNNEncoder(device=device, n_class=7)
    model = DensityGridNN(mpnn_encoder, device=device)

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

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.memory._record_memory_history()

    x = model(atom_grid, classes, edges, batch_nodes)

    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    print(time.perf_counter() - t0)
