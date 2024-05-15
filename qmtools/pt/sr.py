import math

import torch
from torch import nn

from .diffusion import ResBlock


class AtomGrid:

    def __init__(self, pos: torch.Tensor, origin: torch.Tensor, lattice: torch.Tensor, device: str | torch.device = "cpu"):
        self.pos = pos.to(device)  # pos.shape = (n_batch, n_atom, 3)
        self.origin = origin.to(device)  # origin.shape = (n_batch, 3)
        self.lattice = lattice.to(device)  # lattice.shape = (n_batch, 3, 3)
        self.device = device

    def pairwise_distances(self, grid_shape: tuple[int, int, int]) -> torch.Tensor:
        distances = []
        for i_batch in range(self.pos.shape[0]):
            o = self.origin[i_batch]
            l = self.lattice[i_batch].diag()
            grid_xyz = [torch.linspace(o[i], o[i] + l[i], grid_shape[i], device=self.device) for i in range(3)]
            grid = torch.stack(torch.meshgrid(*grid_xyz, indexing="ij"), dim=-1)  # grid.shape = (mx, my, mz, 3)
            grid = grid.reshape(1, -1, 3)  # grid.shape = (1, m_voxel, 3)
            distances.append(torch.cdist(grid, self.pos[i_batch][None], p=2))  # distances.shape = (1, m_voxel, n_atom)
        distances = torch.cat(distances, dim=0)  # distances.shape = (n_batch, m_voxel, n_atom)
        return distances


class SRAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mol_embed_size: int,
        activation: bool = None,
    ):
        super().__init__()

        self.v_linear = nn.Linear(mol_embed_size, in_channels)
        self.out_linear = nn.Linear(2 * in_channels, in_channels)
        self.out_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.act = activation

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor, atom_grid: AtomGrid, batch_nodes: list[int]) -> torch.Tensor:
        # x.shape = (n_batch, c, mx, my, mz)
        # mol_embed.shape = (n_batch, n_atom, mol_embed_size)
        # pos.shape = (n_batch, n_atom, 3)
        # len(batch_nodes) = n_batch

        b, c, mx, my, mz = x.shape
        m_voxel = mx * my * mz
        x_in = x

        # Construct weight matrix based on the pair-wise distances between the atoms and the voxels
        weight = -atom_grid.pairwise_distances((mx, my, mz))  # weight.shape = (n_batch, m_voxel, n_atom)

        # Some of the entries in the molecule embedding are padding due to different size molecule graphs.
        # We set the corresponding entries to have a weight of -inf, so that after softmax the attention weight on
        # those entries is exactly 0.
        for i_batch in range(b):
            weight[i_batch, :, batch_nodes[i_batch] :] = -torch.inf

        att = nn.functional.softmax(weight, dim=-1)  # att.shape = (n_batch, m_voxel, n_atom)

        v = self.v_linear(mol_embed)  # v.shape = (n_batch, n_atom, c)
        v = torch.matmul(att, v)  # x.shape = (n_batch, m_voxel, c)

        x = x.reshape(b, c, m_voxel)  # x.shape = (n_batch, c, m_voxel)
        x = x.transpose(1, 2)  # x.shape = (n_batch, m_voxel, c)
        x = torch.cat([x, v], dim=-1)  # x.shape = (n_batch, m_voxel, 2 * c)
        x = self.act(self.out_linear(x))  # x.shape = (n_batch, m_voxel, c)

        x = x.permute(0, 2, 1)  # x.shape = (n_batch, c, m_voxel)
        x = x.reshape(b, c, mx, my, mz)  # x.shape = (n_batch, c, mx, my, mz)

        x = self.act(self.out_conv(x)) + x_in  # x.shape = (n_batch, c, mx, my, mz)

        return x


class DensitySRDecoder(nn.Module):

    def __init__(self, mol_embed_size: int = 32, device: str | torch.device = "cpu"):
        super().__init__()

        self.act = nn.SiLU()
        self.decoder = nn.Sequential(
            # input_shape = (n_batch, 1, nx / 4, ny / 4, nz / 4)
            ResBlock(1, 64, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            SRAttentionBlock(64, mol_embed_size, activation=self.act),
            # (n_batch, 64, nx / 4, ny / 4, nz / 4)
            nn.Upsample(scale_factor=2, mode="trilinear"),
            # (n_batch, 64, nx / 2, ny / 2, nz / 2)
            ResBlock(64, 32, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            SRAttentionBlock(32, mol_embed_size, activation=self.act),
            # (n_batch, 32, nx / 2, ny / 2, nz / 2)
            nn.Upsample(scale_factor=2, mode="trilinear"),
            # (n_batch, 32, nx, ny, nz)
            ResBlock(32, 16, kernel_size=3, depth=3, padding_mode="circular", activation=self.act),
            # SRAttentionBlock(8, 16, mol_embed_size, activation=self.act),
            # (n_batch, 16, nx, ny, nz)
            nn.Conv3d(16, 1, kernel_size=3, padding=1, padding_mode="circular"),
            # (n_batch, 1, nx, ny, nz)
        )

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor, atom_grid: AtomGrid, batch_nodes: list[int]) -> torch.Tensor:
        # x.shape = (n_batch, nx / 4, ny / 4, nz / 4)
        # mol_embed.shape = (n_batch, n_atom, mol_embed_size)
        # len(batch_nodes) = n_batch

        x = x.unsqueeze(1)  # x.shape = (n_batch, 1, nx / 4, ny / 4, nz / 4)

        for layer in self.decoder:
            if isinstance(layer, SRAttentionBlock):
                x = layer(x, mol_embed, atom_grid, batch_nodes)
            else:
                x = layer(x)

        x = x.squeeze(1)  # x.shape = (n_batch, nx, ny, nz)

        return x
