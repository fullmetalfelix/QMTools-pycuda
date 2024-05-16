import time

import torch

from qmtools.pt.diffusion import DensityDiffusion, DensityDiffusionVAE, MolPotentialEncoder
from qmtools.pt.sr import AtomGrid, DensitySRDecoder
from qmtools.pt.gnn import MPNNEncoder

if __name__ == "__main__":

    device = "cuda"

    vae = DensityDiffusionVAE(device=device)
    diffusion = DensityDiffusion(device=device)
    pot_encoder = MolPotentialEncoder(device=device)

    x = torch.rand(2, 32, 32, 32, device=device)
    pot = torch.rand(2, 2, 32, 32, 32, device=device)
    t = torch.zeros(1, 32, device=device)
    t[0] = 1

    mol_embed = pot_encoder(pot)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    x_mean, x_log_var = vae.encode(x)
    x_latent = vae.sample_latent_space(x_mean, x_log_var)
    x_diffused = diffusion(x_latent, mol_embed, t)
    x_decoded = vae.decode(x_diffused)

    torch.cuda.synchronize()
    print(time.perf_counter() - t0)

    sr_decoder = DensitySRDecoder(device=device)
    mpnn_encoder = MPNNEncoder(device=device, n_class=7)

    n_batch = 2
    x = torch.rand(n_batch, 32, 32, 32, device=device)
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
    atom_grid = AtomGrid(pos, origin, lattice, device=device)
    batch_nodes = [5, 10]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.memory._record_memory_history()

    mol_embed = mpnn_encoder(pos, classes, edges)
    mol_embed, atom_grid = mpnn_encoder.split_graph(mol_embed, atom_grid, batch_nodes)
    x = sr_decoder(x, mol_embed, atom_grid, batch_nodes)

    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    print(time.perf_counter() - t0)
