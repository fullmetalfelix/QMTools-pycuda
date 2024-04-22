import torch
import torch.functional as F
from torch import nn

DIFF_Q = 0.8
ONE_OVER_DIFF_TOT = 0.05234482976098482


class AutomatonPT(nn.Module):

    def __init__(self, n_layer=4):
        super().__init__()
        self.n_layer = n_layer
        layers = []
        for _ in range(self.n_layer):
            layers.append(nn.Linear(16, 16, bias=True))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(16, 1, bias=True))
        self.net = nn.Sequential(*layers)
        self.extra_constants = nn.Parameter(torch.rand(12))

    def forward(self, x):

        # Construct tensor with all neighbour pairs

        # Compute charge transfer for each neighbour pair, symmetrized
        # x.shape = (n_batch, n_voxel, 26, 4)
        b = self.extra_constants.repeat((x.shape[0], x.shape[1], 26, 1))
        x1 = torch.cat([x, b], dim=3)  # x1.shape = (n_batch, n_voxel, 26, 16)
        x2 = torch.cat([x[..., 2:], x[..., :2], b], dim=3)  # x2.shape = (n_batch, n_voxel, 26, 16)
        x = nn.functional.tanh(self.net(x1) - self.net(x2))  # x.shape = (n_batch, n_voxel, 26, 1)
        x = x.squeeze(3)  # x.shape = (n_batch, n_voxel, 26)

        # Get the direction of the transfer right

        # Scale by neighbour distance

        # Sum over neighbours to get total transfer for each voxel
        t = x.sum(dim=2) * (ONE_OVER_DIFF_TOT * DIFF_Q) # t.shape = (n_batch, n_voxel)

        # Add to original charge

        return t
