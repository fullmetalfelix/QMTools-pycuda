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

    def pairwise_similarity(self, scale: torch.Tensor, downscale_factor: int = 1) -> torch.Tensor:
        lorentzians = []
        n_xyz = [s // downscale_factor for s in self.grid_shape]
        for i_batch in range(self.pos.shape[0]):
            o = self.origin[i_batch]
            l = self.lattice[i_batch].diag()
            grid_xyz = [torch.linspace(o[i], o[i] + l[i], n_xyz[i], device=self.device) for i in range(3)]
            grid = torch.stack(torch.meshgrid(*grid_xyz, indexing="ij"), dim=-1)  # grid.shape = (mx, my, mz, 3)
            grid = grid.reshape(1, -1, 3)  # grid.shape = (1, m_voxel, 3)
            distance = torch.cdist(grid, self.pos[i_batch][None], p=2)  # distance.shape = (1, m_voxel, n_atom)
            lorentz = scale / (scale**2 + distance**2)  # lorenz.shape = (1, m_voxel, n_atom)
            lorentzians.append(lorentz)
        lorentzians = torch.cat(lorentzians, dim=0)  # distances.shape = (n_batch, m_voxel, n_atom)
        return lorentzians, n_xyz


class DensityGridNN(nn.Module):

    def __init__(self, mpnn: MPNNEncoder, proj_channels: list[int], device: str | torch.device = "cpu"):
        super().__init__()

        self.mpnn = mpnn
        self.n_stage = len(proj_channels)

        self.node_projections = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        proj_channels.append(1)  # The final feature map only has a single channel corresponding to the electron density
        for stage in range(self.n_stage):
            self.node_projections.append(nn.Linear(mpnn.node_embed_size, proj_channels[stage]))
            self.conv_blocks.append(
                nn.Sequential(
                    ResBlock(
                        proj_channels[stage],
                        proj_channels[stage],
                        kernel_size=3,
                        depth=2,
                        padding_mode="circular",
                        activation=nn.ReLU(),
                    ),
                    nn.Conv3d(proj_channels[stage], proj_channels[stage + 1], kernel_size=1),
                )
            )

        # Separate scale parameter for each upsampling stage
        self.scale = nn.Parameter(0.5 + 0.2 * (torch.rand(self.n_stage) - 0.5))

        self.device = device
        self.to(device)

    def split_graph(self, node_features: torch.Tensor, atom_grid: AtomGrid, batch_nodes: list[int]) -> torch.Tensor:
        # Split combined graph into separate graphs by padding smaller graphs to have the same
        # number of nodes as the biggest graph.
        # node_features.shape: (n_node_total, node_embed_size) -> (n_batch, n_node_biggest, node_embed_size)
        # atom_grid.pos.shape: (n_node_total, 3) -> (n_batch, n_node_biggest, 3)
        node_features = torch.split(node_features, split_size_or_sections=batch_nodes)
        pos = torch.split(atom_grid.pos, split_size_or_sections=batch_nodes)
        max_size = max(batch_nodes)
        node_features_padded = []
        pos_padded = []
        for f, p in zip(node_features, pos):
            assert f.shape[0] == p.shape[0], "Inconsistent node count"
            pad_size = max_size - f.shape[0]
            if pad_size > 0:
                f = torch.cat([f, torch.zeros(pad_size, f.shape[1], device=self.device)], dim=0)
                p = torch.cat([p, torch.zeros(pad_size, p.shape[1], device=self.device)], dim=0)
            node_features_padded.append(f)
            pos_padded.append(p)
        node_features = torch.stack(node_features_padded, axis=0)
        atom_grid.pos = torch.stack(pos_padded, axis=0)
        return node_features, atom_grid

    def project_onto_grid(self, stage: int, node_features: torch.Tensor, atom_grid: AtomGrid, batch_nodes: list[int]):

        n_batch = len(batch_nodes)

        # We use the pairwise distances between each voxel and atom as weights for projecting the node
        # embeddings onto a grid.
        downscale_factor = 2 ** (self.n_stage - stage - 1)
        weight, (nx, ny, nz) = atom_grid.pairwise_similarity(scale=self.scale[stage], downscale_factor=downscale_factor)
        # weight.shape = (n_batch, n_voxel, n_atom), n_voxel = nx * ny * nz

        # Some of the entries in the molecule embedding are padding due to different size molecule graphs.
        # We set the corresponding entries to have a weight of 0
        for i_batch in range(n_batch):
            weight[i_batch, :, batch_nodes[i_batch] :] = 0

        # Project the node embeddings onto the grid.
        node_features = self.node_projections[stage](node_features)  # node_features.shape = (n_batch, n_atom, c_proj)
        x = torch.matmul(weight, node_features)  # x.shape = (n_batch, n_voxel, c_proj)
        x = x.transpose(1, 2)  # x.shape = (n_batch, c_proj, n_voxel)
        x = x.reshape(n_batch, x.shape[1], nx, ny, nz)  # x.shape = (n_batch, c_proj, nx, ny, nz)

        return x

    def forward(self, atom_grid: AtomGrid, classes: torch.Tensor, edges: torch.Tensor, batch_nodes: list[int]) -> torch.Tensor:

        node_features = self.mpnn(atom_grid.pos, classes, edges)
        node_features, atom_grid = self.split_graph(node_features, atom_grid, batch_nodes)
        # node_features.shape = (n_batch, n_atom, c_node)
        # atom_grid.pos.shape = (n_batch, n_atom, 3)

        # print(self.scale)

        x = None
        for stage in range(self.n_stage):

            # Project the node embeddings onto a grid
            x_grid = self.project_onto_grid(stage, node_features, atom_grid, batch_nodes)
            # x_grid.shape = (n_batch, c_proj, nx, ny, nz)

            if stage == 0:
                # On the first stage we simply use the projection as the feature map
                x = x_grid
            else:
                # After first stage the grid projection is added to the previous feature map. The feature map
                # from the previous stage needs to be upscaled in order to be of the same shape as the current
                # projection.
                # x.shape: (n_batch, c_proj, nx / 2, ny / 2, nz / 2) -> (n_batch, c_proj, nx, ny, nz)
                x = nn.functional.interpolate(x, size=x_grid.shape[2:], mode="trilinear", align_corners=False)
                x = x + x_grid

            x = self.conv_blocks[stage](x)  # x.shape = (n_batch, c_proj_next, nx, ny, nz)

        x = x.squeeze(1)  # x.shape = (n_batch, nx, ny, nz)

        return x
