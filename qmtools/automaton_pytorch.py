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
