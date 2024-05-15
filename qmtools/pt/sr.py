import math

import torch
from torch import nn

from .diffusion import ResBlock


class SRAttentionBlock(nn.Module):
    def __init__(
        self,
        n_head: int,
        in_channels: int,
        mol_embed_size: int,
        activation: bool = None,
    ):
        super().__init__()

        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(mol_embed_size, in_channels)
        self.v_linear = nn.Linear(mol_embed_size, in_channels)
        self.out_linear = nn.Linear(in_channels, in_channels)
        self.out_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.n_head = n_head
        self.head_size = in_channels // n_head
        assert self.head_size * n_head == in_channels, "Number of channels must be divisible by the number of heads"
        self.act = activation

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor, batch_nodes: list[int]) -> torch.Tensor:
        # x.shape = (n_batch, c, mx, my, mz)
        # mol_embed.shape = (n_batch, n_atom, mol_embed_size)
        # len(batch_nodes) = n_batch

        b, c, mx, my, mz = x.shape
        m_voxel = mx * my * mz
        n_atom = mol_embed.shape[1]

        x_in = x

        x = x.reshape(b, c, m_voxel)  # x.shape = (n_batch, c, m_voxel)
        x = x.transpose(1, 2)  # x.shape = (n_batch, m_voxel, c)

        q = self.q_linear(x)  # q.shape = (n_batch, m_voxel, c)
        k = self.k_linear(mol_embed)  # k.shape = (n_batch, n_atom, c)
        v = self.v_linear(mol_embed)  # v.shape = (n_batch, n_atom, c)

        q = q.reshape(b, m_voxel, self.n_head, self.head_size)  # q.shape = (n_batch, m_voxel, n_head, c / n_head)
        k = k.reshape(b, n_atom, self.n_head, self.head_size)  # k.shape = (n_batch, n_atom, n_head, c / n_head)
        v = v.reshape(b, n_atom, self.n_head, self.head_size)  # v.shape = (n_batch, n_atom, n_head, c / n_head)

        att = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))  # att.shape = (n_batch, n_head, m_voxel, n_atom)

        # Some of the entries in the molecule embedding are padding due to different size molecule graphs.
        # We set the corresponding entries to have a weight of -inf, so that after softmax the attention weight on
        # those entries is exactly 0.
        for i_batch in range(b):
            att[i_batch, :, :, batch_nodes[i_batch] :] = -torch.inf

        att /= math.sqrt(self.head_size)
        att = nn.functional.softmax(att, dim=-1)

        x = torch.matmul(att, v.permute(0, 2, 1, 3))  # x.shape = (n_batch, n_head, m_voxel, c / n_head)
        x = x.permute((0, 2, 1, 3))  # x.shape = (n_batch, m_voxel, n_head, c / n_head)
        x = x.reshape(b, m_voxel, c)  # x.shape = (n_batch, m_voxel, c)

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
            SRAttentionBlock(8, 64, mol_embed_size, activation=self.act),
            # (n_batch, 64, nx / 4, ny / 4, nz / 4)
            nn.Upsample(scale_factor=2, mode="trilinear"),
            # (n_batch, 64, nx / 2, ny / 2, nz / 2)
            ResBlock(64, 32, kernel_size=3, depth=2, padding_mode="circular", activation=self.act),
            SRAttentionBlock(8, 32, mol_embed_size, activation=self.act),
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

    def forward(self, x: torch.Tensor, mol_embed: torch.Tensor, batch_nodes: list[int]) -> torch.Tensor:
        # x.shape = (n_batch, nx / 4, ny / 4, nz / 4)
        # mol_embed.shape = (n_batch, n_atom, mol_embed_size)
        # len(batch_nodes) = n_batch

        x = x.unsqueeze(1)  # x.shape = (n_batch, 1, nx / 4, ny / 4, nz / 4)

        for layer in self.decoder:
            if isinstance(layer, SRAttentionBlock):
                x = layer(x, mol_embed, batch_nodes)
            else:
                x = layer(x)

        x = x.squeeze(1)  # x.shape = (n_batch, nx, ny, nz)

        return x
