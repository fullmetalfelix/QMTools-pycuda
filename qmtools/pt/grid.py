from copy import deepcopy
import math
import torch
from torch import nn

from qmtools.pt.diffusion import ResBlock
from qmtools.pt.gnn import MPNNEncoder


class AtomGrid:

    def __init__(
        self,
        pos: torch.Tensor,
        origin: torch.Tensor,
        lattice: torch.Tensor,
        grid_shape: tuple[int, int, int],
        device: str | torch.device = "cpu",
    ):
        self.pos = pos.to(device)  # pos.shape = (n_batch, n_atom, 3)
        self.origin = origin.to(device)  # origin.shape = (n_batch, 3)
        self.lattice = lattice.to(device)  # lattice.shape = (n_batch, 3, 3)
        self.grid_shape = grid_shape
        self.device = device


class Lorentzian(torch.autograd.Function):

    @staticmethod
    def forward(ctx, distance: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # distance.shape = (n_batch, n_voxel, n_atom, 1)
        # scale.shape = (1, 1, 1, c)
        lorentz = scale / (scale * scale + distance * distance)  # lorenz.shape = (n_batch, n_voxel, n_atom, c)
        ctx.save_for_backward(distance, scale, lorentz)
        return lorentz

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        distance, scale, lorentz = ctx.saved_tensors
        # distance.shape = (n_batch, n_voxel, n_atom, 1)
        # scale.shape = (1, 1, 1, c)
        # lorentz.shape = (n_batch, n_voxel, n_atom, c)
        grad_distance = grad_scale = None
        s_inv = 1 / scale
        if ctx.needs_input_grad[0]:
            grad_distance = grad_output * -2 * distance * s_inv * lorentz * lorentz
        if ctx.needs_input_grad[1]:
            grad_scale = s_inv - 2 * lorentz
            grad_scale *= lorentz * grad_output  # Slightly reduced memory usage from doing this in-place
        return grad_distance, grad_scale


class Lorentzian2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, distance: torch.Tensor, scale: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        # distance.shape = (n_batch, n_voxel, n_atom, 1)
        # scale.shape = (1, 1, 1, c)
        # amplitude.shape = (1, 1, 1, c)
        lorentz = amplitude / (scale * scale + distance * distance)  # lorenz.shape = (n_batch, n_voxel, n_atom, c)
        ctx.save_for_backward(distance, scale, amplitude, lorentz)
        return lorentz

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distance, scale, amplitude, lorentz = ctx.saved_tensors
        # distance.shape = (n_batch, n_voxel, n_atom, 1)
        # scale.shape = (1, 1, 1, c)
        # amplitude.shape = (1, 1, 1, c)
        # lorentz.shape = (n_batch, n_voxel, n_atom, c)
        # grad_output.shape = (n_batch, n_voxel, n_atom, c)
        grad_distance = grad_scale = grad_amplitude = None
        a_inv = 1 / amplitude
        if ctx.needs_input_grad[0]:
            grad_distance = -2 * a_inv * grad_output
            grad_distance *= distance
            grad_distance *= lorentz
            grad_distance *= lorentz
        if ctx.needs_input_grad[1]:
            grad_scale = -2 * scale * a_inv * grad_output
            grad_scale *= lorentz
            grad_scale *= lorentz
        if ctx.needs_input_grad[2]:
            grad_amplitude = grad_output.clone()
            grad_amplitude *= a_inv
            grad_amplitude *= lorentz
        return grad_distance, grad_scale, grad_amplitude


lorentzian = Lorentzian.apply
lorentzian2 = Lorentzian2.apply


class DensityGridNN(nn.Module):

    def __init__(
        self,
        mpnn: MPNNEncoder,
        proj_channels: list[int],
        cnn_channels: list[int],
        lorentz_type: int = 1,
        per_channel_scale: bool = False,
        scale_init_bounds: tuple[int, int] = (0.5, 1.5),
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        assert len(proj_channels) == len(cnn_channels)

        self.mpnn = mpnn
        self.n_stage = len(proj_channels)

        self.node_projections = nn.ModuleList([])
        self.proj_convs = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        cnn_channels.append(1)  # The final feature map only has a single channel corresponding to the electron density
        for stage in range(self.n_stage):
            self.node_projections.append(nn.Linear(mpnn.node_embed_size, proj_channels[stage]))
            self.proj_convs.append(nn.Conv3d(proj_channels[stage], cnn_channels[stage], kernel_size=1))
            self.conv_blocks.append(
                nn.Sequential(
                    ResBlock(
                        cnn_channels[stage],
                        cnn_channels[stage],
                        kernel_size=3,
                        depth=2,
                        padding_mode="circular",
                        activation=nn.ReLU(),
                    ),
                    nn.Conv3d(cnn_channels[stage], cnn_channels[stage + 1], kernel_size=1),
                )
            )

        self.lorentz_type = lorentz_type
        if lorentz_type == 1:
            a = scale_init_bounds[0]
            b = 1 / a
            self.amplitude = None
            if per_channel_scale:
                # Separate scale parameter for each channel in every upsampling stage
                self.scale = nn.ParameterList([nn.Parameter(a + (b - a) * torch.rand(c)) for c in proj_channels])
            else:
                # Separate scale parameter for each upsampling stage
                self.scale = nn.Parameter(a + (b - a) * torch.rand(self.n_stage))
        else:
            c, d = scale_init_bounds
            a = 3 * math.sqrt(c**3 * d**3 / (c**2 + c * d + d**2))
            self.amplitude = nn.ParameterList([nn.Parameter(2 * a * (torch.rand(n_ch) - 0.5)) for n_ch in proj_channels])
            self.scale = nn.ParameterList([nn.Parameter(c + (d - c) * torch.rand(n_ch)) for n_ch in proj_channels])

        self.device = device
        self.to(device)

    def split_graph(self, node_features: torch.Tensor, atom_grid: AtomGrid, batch_nodes: list[int]) -> torch.Tensor:
        # Split combined graph into separate graphs by padding smaller graphs to have the same
        # number of nodes as the biggest graph.
        # node_features.shape: (n_node_total, node_embed_size) -> (n_batch, n_node_biggest, node_embed_size)
        # atom_grid.pos.shape: (n_node_total, 3) -> (n_batch, n_node_biggest, 3)
        dtype = node_features.dtype
        node_features = torch.split(node_features, split_size_or_sections=batch_nodes)
        pos = torch.split(atom_grid.pos, split_size_or_sections=batch_nodes)
        max_size = max(batch_nodes)
        node_features_padded = []
        pos_padded = []
        for f, p in zip(node_features, pos):
            assert f.shape[0] == p.shape[0], "Inconsistent node count"
            pad_size = max_size - f.shape[0]
            if pad_size > 0:
                f = torch.cat([f, torch.zeros(pad_size, f.shape[1], device=self.device, dtype=dtype)], dim=0)
                p = torch.cat([p, torch.zeros(pad_size, p.shape[1], device=self.device)], dim=0)
            node_features_padded.append(f)
            pos_padded.append(p)
        node_features = torch.stack(node_features_padded, axis=0)
        atom_grid = deepcopy(atom_grid)
        atom_grid.pos = torch.stack(pos_padded, axis=0)
        return node_features, atom_grid

    def pairwise_similarity(
        self, atom_grid: AtomGrid, stage: int, dtype: torch.dtype, batch_nodes: list[int]
    ) -> tuple[torch.Tensor, list[int]]:

        downscale_factor = 2 ** (self.n_stage - stage - 1)
        n_xyz = [s // downscale_factor for s in atom_grid.grid_shape]
        l = atom_grid.lattice[0].diag()  # The lattice is assumed to be the same size for all batch items here

        grid_xyz = [torch.linspace(0, l[i], n_xyz[i], device=self.device) for i in range(3)]
        grid = torch.stack(torch.meshgrid(*grid_xyz, indexing="ij"), dim=-1)  # grid.shape = (nx, ny, nz, 3)
        grid = grid.reshape(1, -1, 3)  # grid.shape = (1, n_voxel, 3)
        grid = grid + atom_grid.origin[:, None, :]  # grid.shape = (n_batch, n_voxel, 3)

        distance = torch.cdist(grid, atom_grid.pos, p=2)  # distance.shape = (n_batch, n_voxel, n_atom)
        distance = distance[:, :, :, None]  # distance.shape = (n_batch, n_voxel, n_atom, 1)
        distance = distance.type(dtype)

        # Some of the entries in the molecule embedding are padding due to different size molecule graphs.
        # We set the corresponding entries to have a distance of inf, so that the weight will be 0.
        for i_batch, n_node in enumerate(batch_nodes):
            distance[i_batch, :, n_node:, :] = torch.inf

        scale = self.scale[stage].type(dtype)
        if scale.shape != ():
            scale = scale[None, None, None]  # scale.shape = (1, 1, 1, c)
        else:
            scale = scale[None, None, None, None]  # scale.shape = (1, 1, 1, 1) (c=1)

        if self.lorentz_type == 1:
            lorentz = lorentzian(distance, scale)  # lorenz.shape = (n_batch, n_voxel, n_atom, c)
        else:
            amplitude = self.amplitude[stage].type(dtype)
            amplitude = amplitude[None, None, None]  # amplitude.shape = (1, 1, 1, c)
            lorentz = lorentzian2(distance, scale, amplitude)

        return lorentz, n_xyz

    def project_onto_grid(self, stage: int, node_features: torch.Tensor, atom_grid: AtomGrid, batch_nodes: list[int]):

        n_batch = len(batch_nodes)

        # We use the pairwise distances between each voxel and atom as weights for projecting the node
        # embeddings onto a grid.
        weight, (nx, ny, nz) = self.pairwise_similarity(atom_grid, stage, node_features.dtype, batch_nodes)
        # weight.shape = (n_batch, n_voxel, n_atom, c_proj), n_voxel = nx * ny * nz

        # Project the node embeddings onto the grid.
        node_features = self.node_projections[stage](node_features)  # node_features.shape = (n_batch, n_atom, c_proj)
        node_features = node_features.unsqueeze(1)  # node_features.shape = (n_batch, 1, n_atom, c_proj)
        x = (weight * node_features).sum(dim=2)  # x.shape = (n_batch, n_voxel, c_proj)
        x = x.transpose(1, 2)  # x.shape = (n_batch, c_proj, n_voxel)
        x = x.reshape(n_batch, x.shape[1], nx, ny, nz)  # x.shape = (n_batch, c_proj, nx, ny, nz)

        x = self.proj_convs[stage](x)  # x.shape = (n_batch, c_cnn, nx, ny, nz)

        return x

    def forward(self, atom_grid: AtomGrid, classes: torch.Tensor, edges: torch.Tensor, batch_nodes: list[int]) -> torch.Tensor:

        node_features = self.mpnn(atom_grid.pos, classes, edges)
        node_features, atom_grid = self.split_graph(node_features, atom_grid, batch_nodes)
        # node_features.shape = (n_batch, n_atom, c_node)
        # atom_grid.pos.shape = (n_batch, n_atom, 3)

        x = None
        for stage in range(self.n_stage):

            # Project the node embeddings onto a grid
            x_grid = self.project_onto_grid(stage, node_features, atom_grid, batch_nodes)
            # x_grid.shape = (n_batch, c, nx, ny, nz)

            if stage == 0:
                # On the first stage we simply use the projection as the feature map
                x = x_grid
            else:
                # After first stage the grid projection is added to the previous feature map. The feature map
                # from the previous stage needs to be upscaled in order to be of the same shape as the current
                # projection.
                # x.shape: (n_batch, c, nx / 2, ny / 2, nz / 2) -> (n_batch, c, nx, ny, nz)
                x = nn.functional.interpolate(x, size=x_grid.shape[2:], mode="trilinear", align_corners=False)
                x = x + x_grid

            x = self.conv_blocks[stage](x)  # x.shape = (n_batch, c_next, nx, ny, nz)

        x = x.squeeze(1)  # x.shape = (n_batch, nx, ny, nz)

        return x


def fd_grad(arr: torch.Tensor):
    """Compute finite-difference gradient by five-point stencil"""
    # arr.shape = (n_batch, nx, ny, nz)
    arr = nn.functional.pad(arr, pad=(2, 2, 2, 2, 2, 2), mode="circular")  # arr.shape = (n_batch, nx + 4, ny + 4, nz + 4)
    # fmt: off
    grad_x = (arr[:,  :-4, 2:-2, 2:-2] - 8 * arr[:, 1:-3, 2:-2, 2:-2] + 8 * arr[:, 3:-1, 2:-2, 2:-2] - arr[:, 4:  , 2:-2, 2:-2]) / 12
    grad_y = (arr[:, 2:-2,  :-4, 2:-2] - 8 * arr[:, 2:-2, 1:-3, 2:-2] + 8 * arr[:, 2:-2, 3:-1, 2:-2] - arr[:, 2:-2, 4:  , 2:-2]) / 12
    grad_z = (arr[:, 2:-2, 2:-2,  :-4] - 8 * arr[:, 2:-2, 2:-2, 1:-3] + 8 * arr[:, 2:-2, 2:-2, 3:-1] - arr[:, 2:-2, 2:-2, 4:  ]) / 12
    # fmt: on
    grad = torch.stack((grad_x, grad_y, grad_z), dim=-1)  # grad.shape = (n_batch, nx, ny, nz, 3)
    return grad


class GridLoss(nn.Module):

    def __init__(self, grad_factor: float = 1.0):
        super().__init__()
        self.grad_factor = grad_factor

    def forward(self, pred: torch.Tensor, ref: torch.Tensor):
        mse = nn.functional.mse_loss(pred, ref)
        mse_grad = nn.functional.mse_loss(fd_grad(pred), fd_grad(ref))
        loss = mse + self.grad_factor * mse_grad
        return [loss, mse, mse_grad]
