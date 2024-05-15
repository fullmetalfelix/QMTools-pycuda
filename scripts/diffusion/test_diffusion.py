import time

import torch

from qmtools.pt.diffusion import DensityDiffusion, DensityDiffusionVAE, MolPotentialEncoder
from qmtools.pt.sr import DensitySRDecoder
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

    x = torch.rand(2, 16, 16, 16, device=device)
    pos = torch.rand(15, 3, device=device)
    classes = torch.rand(15, 7, device=device)
    edges = torch.tensor(
        [
            [0, 1, 4, 6, 2, 4, 5, 2],
            [5, 14, 8, 2, 5, 3, 4, 12],
        ],
        device=device,
    )
    batch_nodes = [5, 10]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    mol_embed = mpnn_encoder(pos, classes, edges)
    mol_embed = mpnn_encoder.split_graph(mol_embed, batch_nodes)
    x = sr_decoder(x, mol_embed, batch_nodes)

    torch.cuda.synchronize()
    print(time.perf_counter() - t0)
