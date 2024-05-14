from typing import Tuple

import torch
from torch import nn

DIFF_Q = 0.8
ONE_OVER_DIFF_TOT = 0.05234482976098482


def _make_inverse_pair_distance_vector():
    d_nearest = 1
    d_edge = 2**0.5
    d_corner = 3**0.5
    d = torch.tensor([d_nearest] * 6 + [d_edge] * 12 + [d_corner] * 8)
    d_inv = 1 / d
    return d_inv


def make_pairs(q):

    # Do periodic padding of the grid
    q_padded = q.detach().clone()  # q_padded.shape = (n_batch, nx, ny, nz, 2)
    q_padded = torch.cat([q_padded[:, -1:, :, :, :], q_padded, q_padded[:, :1, :, :, :]], dim=1)  # (n_batch, nx+2, ny, nz, 2)
    q_padded = torch.cat([q_padded[:, :, -1:, :, :], q_padded, q_padded[:, :, :1, :, :]], dim=2)  # (n_batch, nx+2, ny+2, nz, 2)
    q_padded = torch.cat([q_padded[:, :, :, -1:, :], q_padded, q_padded[:, :, :, :1, :]], dim=3)  # (n_batch, nx+2, ny+2, nz+2, 2)

    # Construct all neighbour pairs
    # fmt: off
    q = torch.stack(
        [
            torch.cat([q, q_padded[:, 2:  , 1:-1, 1:-1]], dim=-1),  # +1 in x
            torch.cat([q, q_padded[:,  :-2, 1:-1, 1:-1]], dim=-1),  # -1 in x
            torch.cat([q, q_padded[:, 1:-1, 2:  , 1:-1]], dim=-1),  # +1 in y
            torch.cat([q, q_padded[:, 1:-1,  :-2, 1:-1]], dim=-1),  # -1 in y
            torch.cat([q, q_padded[:, 1:-1, 1:-1, 2:  ]], dim=-1),  # +1 in z
            torch.cat([q, q_padded[:, 1:-1, 1:-1,  :-2]], dim=-1),  # -1 in z
            torch.cat([q, q_padded[:, 2:  , 2:  , 1:-1]], dim=-1),  # +1 in x, +1 in y
            torch.cat([q, q_padded[:,  :-2, 2:  , 1:-1]], dim=-1),  # -1 in x, +1 in y
            torch.cat([q, q_padded[:, 2:  ,  :-2, 1:-1]], dim=-1),  # +1 in x, -1 in y
            torch.cat([q, q_padded[:,  :-2,  :-2, 1:-1]], dim=-1),  # -1 in x, -1 in y
            torch.cat([q, q_padded[:, 2:  , 1:-1, 2:  ]], dim=-1),  # +1 in x, +1 in z
            torch.cat([q, q_padded[:,  :-2, 1:-1, 2:  ]], dim=-1),  # -1 in x, +1 in z
            torch.cat([q, q_padded[:, 2:  , 1:-1,  :-2]], dim=-1),  # +1 in x, -1 in z
            torch.cat([q, q_padded[:,  :-2, 1:-1,  :-2]], dim=-1),  # -1 in x, -1 in z
            torch.cat([q, q_padded[:, 1:-1, 2:  , 2:  ]], dim=-1),  # +1 in y, +1 in z
            torch.cat([q, q_padded[:, 1:-1,  :-2, 2:  ]], dim=-1),  # -1 in y, +1 in z
            torch.cat([q, q_padded[:, 1:-1, 2:  ,  :-2]], dim=-1),  # +1 in y, -1 in z
            torch.cat([q, q_padded[:, 1:-1,  :-2,  :-2]], dim=-1),  # -1 in y, -1 in z
            torch.cat([q, q_padded[:, 2:  , 2:  , 2:  ]], dim=-1),  # +1 in x, +1 in y, +1 in z
            torch.cat([q, q_padded[:, 2:  , 2:  ,  :-2]], dim=-1),  # +1 in x, +1 in y, -1 in z
            torch.cat([q, q_padded[:, 2:  ,  :-2, 2:  ]], dim=-1),  # +1 in x, -1 in y, +1 in z
            torch.cat([q, q_padded[:, 2:  ,  :-2,  :-2]], dim=-1),  # +1 in x, -1 in y, -1 in z
            torch.cat([q, q_padded[:,  :-2, 2:  , 2:  ]], dim=-1),  # -1 in x, +1 in y, +1 in z
            torch.cat([q, q_padded[:,  :-2, 2:  ,  :-2]], dim=-1),  # -1 in x, +1 in y, -1 in z
            torch.cat([q, q_padded[:,  :-2,  :-2, 2:  ]], dim=-1),  # -1 in x, -1 in y, +1 in z
            torch.cat([q, q_padded[:,  :-2,  :-2,  :-2]], dim=-1),  # -1 in x, -1 in y, -1 in z
        ],
        dim=4,
    )  # fmt: on

    return q  # q.shape = (n_batch, nx, ny, nz, 26, 4)


def _get_padding(kernel_size: int | Tuple[int, ...], nd: int) -> Tuple[int, ...]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    padding = []
    for i in range(nd):
        padding += [(kernel_size[i] - 1) // 2]
    return tuple(padding)


class Conv3dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, ...] = 3,
        depth: int = 2,
        padding_mode: str = "zeros",
        res_connection: bool = True,
        activation: bool = None,
        last_activation: bool = True,
    ):
        assert depth >= 1

        super().__init__()

        self.res_connection = res_connection
        if not activation:
            self.act = nn.ReLU()
        else:
            self.act = activation

        if last_activation:
            self.acts = [self.act] * depth
        else:
            self.acts = [self.act] * (depth - 1) + [self._identity]

        padding = _get_padding(kernel_size, 3)
        self.convs = nn.ModuleList(
            [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)]
        )
        for _ in range(depth - 1):
            self.convs.append(
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
            )
        if res_connection and in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def _identity(self, x):
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x))
        if self.res_connection:
            if self.res_conv:
                x = x + self.res_conv(x_in)
            else:
                x = x + x_in
        return x


class AutomatonPT(nn.Module):

    def __init__(self, n_layer=4, device="cpu"):

        super().__init__()
        self.n_layer = n_layer
        self.inverse_pair_distances = _make_inverse_pair_distance_vector().to(device)
        self.device = device

        layers = [
            nn.Linear(4, 16, bias=True),
            nn.Tanh(),
        ]
        for _ in range(self.n_layer - 1):
            layers.append(nn.Linear(16, 16, bias=True))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(16, 1, bias=True))

        self.net = nn.Sequential(*layers)

        self.to(device)

    def forward(self, q):

        # q.shape = (n_batch, nx, ny, nz, 2)

        # Construct tensor with all neighbour pairs
        q_pairs = make_pairs(q)  # q_pairs.shape = (n_batch, nx, ny, nz, 26, 4)

        # Compute charge transfer for each neighbour pair, symmetrized
        x1 = q_pairs
        x2 = torch.cat([q_pairs[..., 2:], q_pairs[..., :2]], dim=5)
        transfer = nn.functional.tanh(self.net(x1) - self.net(x2))  # transfer.shape = (n_batch, nx, ny, nz, 26, 1)
        transfer = transfer.squeeze(5)  # transfer.shape = (n_batch, nx, ny, nz, 26)

        # Get the direction of the transfer right
        mask_neg_transfer = transfer < 0
        mask_pos_transfer = ~mask_neg_transfer
        transfer = transfer.clone()
        transfer[mask_neg_transfer] *= q_pairs[mask_neg_transfer][:, 0]
        transfer[mask_pos_transfer] *= q_pairs[mask_pos_transfer][:, 2]

        # Scale by neighbour distances
        transfer = transfer * self.inverse_pair_distances[None, None, None, None, :]

        # Sum over neighbours to get total transfer for each voxel
        transfer = transfer.sum(dim=(4)) * (ONE_OVER_DIFF_TOT * DIFF_Q)  # transfer.shape = (n_batch, nx, ny, nz)

        # Add to original charge
        q_updated = q.detach().clone()
        q_updated[..., 0] += transfer

        return q_updated


class DensityCNN(nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()

        self.act = nn.ReLU()
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")

        self.conv_enc1 = Conv3dBlock(2, 8, depth=2, padding_mode="circular", activation=self.act)
        self.conv_enc2 = Conv3dBlock(8, 16, depth=2, padding_mode="circular", activation=self.act)
        self.conv_enc3 = Conv3dBlock(16, 32, depth=2, padding_mode="circular", activation=self.act)
        self.conv_middle = Conv3dBlock(32, 32, depth=2, padding_mode="circular", activation=self.act)
        self.conv_dec1 = Conv3dBlock(64, 16, depth=2, padding_mode="circular", activation=self.act)
        self.conv_dec2 = Conv3dBlock(32, 8, depth=2, padding_mode="circular", activation=self.act)
        self.conv_dec3 = Conv3dBlock(16, 8, depth=2, padding_mode="circular", activation=self.act)
        self.conv_final = Conv3dBlock(8, 1, depth=2, padding_mode="circular", activation=self.act)

        self.device = device
        self.to(device)

    def forward(self, q):

        # q.shape = (n_batch, 2, nx, ny, nz)

        q_conv_enc1 = self.conv_enc1(q)  # (n_batch, 8, nx, ny, nz)
        q_conv_enc2 = self.conv_enc2(self.pool(q_conv_enc1))  # (n_batch, 16, nx / 2, ny / 2, nz / 2)
        q_conv_enc3 = self.conv_enc3(self.pool(q_conv_enc2))  # (n_batch, 32, nx / 4, ny / 4, nz / 4)

        q_conv_middle = self.conv_middle(self.pool(q_conv_enc3))  # (n_batch, 32, nx / 8, ny / 8, nz / 8)

        q_comb1 = torch.cat([self.upsample(q_conv_middle), q_conv_enc3], dim=1)  # (n_batch, 64, nx / 4, ny / 4, nz / 4)
        q_conv_dec1 = self.conv_dec1(q_comb1)  # (n_batch, 16, nx / 4, ny / 4, nz / 4)
        q_comb2 = torch.cat([self.upsample(q_conv_dec1), q_conv_enc2], dim=1)  # (n_batch, 32, nx / 2, ny / 2, nz / 2)
        q_conv_dec2 = self.conv_dec2(q_comb2)  # (n_batch, 8, nx / 2, ny / 2, nz / 2)
        q_comb3 = torch.cat([self.upsample(q_conv_dec2), q_conv_enc1], dim=1)  # (n_batch, 16, nx / 2, ny / 2, nz / 2)
        q_conv_dec3 = self.conv_dec3(q_comb3)  # (n_batch, 8, nx, ny, nz)

        q_final = self.conv_final(q_conv_dec3)  # (n_batch, 1, nx, ny, nz)
        q_final = q_final.squeeze(1)  # (n_batch, nx, ny, nz)

        return q_final
