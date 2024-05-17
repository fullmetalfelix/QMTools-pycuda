import torch
from torch import nn

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

    def pairwise_similarity(self, scale: torch.Tensor) -> torch.Tensor:
        lorentzians = []
        for i_batch in range(self.pos.shape[0]):
            o = self.origin[i_batch]
            l = self.lattice[i_batch].diag()
            grid_xyz = [torch.linspace(o[i], o[i] + l[i], self.grid_shape[i], device=self.device) for i in range(3)]
            grid = torch.stack(torch.meshgrid(*grid_xyz, indexing="ij"), dim=-1)  # grid.shape = (mx, my, mz, 3)
            grid = grid.reshape(1, -1, 3)  # grid.shape = (1, m_voxel, 3)
            distance = torch.cdist(grid, self.pos[i_batch][None], p=2)  # distance.shape = (1, m_voxel, n_atom)
            lorentz = scale / (scale**2 + distance**2)  # lorenz.shape = (1, m_voxel, n_atom)
            lorentzians.append(lorentz)
        lorentzians = torch.cat(lorentzians, dim=0)  # distances.shape = (n_batch, m_voxel, n_atom)
        return lorentzians


class DensityGridNN(nn.Module):

    def __init__(self, mpnn: MPNNEncoder, device: str | torch.device = "cpu"):
        super().__init__()

        self.mpnn = mpnn
        self.decoder = nn.Sequential(
            nn.Linear(mpnn.node_embed_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.5))

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

    def forward(self, atom_grid: AtomGrid, classes: torch.Tensor, edges: torch.Tensor, batch_nodes: list[int]) -> torch.Tensor:

        n_batch = len(batch_nodes)
        nx, ny, nz = atom_grid.grid_shape

        node_features = self.mpnn(atom_grid.pos, classes, edges)
        node_features, atom_grid = self.split_graph(node_features, atom_grid, batch_nodes)
        # node_features.shape = (n_batch, n_atom, c)
        # atom_grid.pos.shape = (n_batch, n_atom, 3)

        # We use the pairwise distances between each voxel and atom as weights for projecting the node
        # embeddings onto a grid.
        print(self.scale)
        weight = atom_grid.pairwise_similarity(scale=self.scale)  # weight.shape = (n_batch, n_voxel, n_atom)

        # Some of the entries in the molecule embedding are padding due to different size molecule graphs.
        # We set the corresponding entries to have a weight of 0
        for i_batch in range(n_batch):
            weight[i_batch, :, batch_nodes[i_batch] :] = 0

        # Project the node embeddings onto the grid.
        x = torch.matmul(weight, node_features)  # x.shape = (n_batch, n_voxel, c)
        c = x.reshape(n_batch, nx, ny, nz, x.shape[-1])

        # Predict density at each grid point
        x = self.decoder(x)  # x.shape = (n_batch, n_voxel, 1)
        x = x.squeeze(-1)  # x.shape = (n_batch, n_voxel)
        x = x.reshape(n_batch, nx, ny, nz)  # x.shape = (n_batch, nx, ny, nz)

        return x, c
