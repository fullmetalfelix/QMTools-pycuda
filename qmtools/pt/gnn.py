import numpy as np
import torch
from torch import nn

CLASSES = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35]

# Reference: http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
# fmt:off
BOND_LENGTHS = {
     1: { 1: 0.74,  6: 1.09,  7: 1.01,  8: 0.96,  9: 0.92, 14: 1.48, 15: 1.42, 16: 1.34, 17: 1.27, 35: 1.41, 53: 1.61},
     6: {           6: 1.54,  7: 1.47,  8: 1.43,  9: 1.33, 14: 1.86, 15: 1.87, 16: 1.81, 17: 1.77, 35: 1.94, 53: 2.13},
     7: {                     7: 1.46,  8: 1.44,  9: 1.39, 14: 1.72, 15: 1.77, 16: 1.68, 17: 1.91, 35: 2.14, 53: 2.22},
     8: {                               8: 1.48,  9: 1.42, 14: 1.61, 15: 1.60, 16: 1.51, 17: 1.64, 35: 1.72, 53: 1.94},
     9: {                                         9: 1.43, 14: 1.56, 15: 1.56, 16: 1.58, 17: 1.66, 35: 1.78, 53: 1.87},
    14: {                                                  14: 2.34, 15: 2.27, 16: 2.10, 17: 2.04, 35: 2.16, 53: 2.40},
    15: {                                                            15: 2.21, 16: 2.10, 17: 2.04, 35: 2.22, 53: 2.43},
    16: {                                                                      16: 2.04, 17: 2.01, 35: 2.25, 53: 2.34},
    17: {                                                                                17: 1.99, 35: 2.18, 53: 2.43},
    35: {                                                                                          35: 2.28, 53: 2.48},
    53: {                                                                                                    53: 2.66}
}
# fmt: on


class MPNNEncoder(nn.Module):

    def __init__(
        self,
        n_class: int,
        iters: int = 6,
        node_embed_size: int = 32,
        hidden_size: int = 32,
        message_size: int = 32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.iters = iters
        self.message_size = message_size
        self.node_embed_size = node_embed_size

        self.act = nn.ReLU()
        self.in_linear = nn.Linear(n_class, node_embed_size)
        self.msg_net = nn.Sequential(
            nn.Linear(2 * node_embed_size + 3, hidden_size),
            self.act,
            nn.Linear(hidden_size, hidden_size),
            self.act,
            nn.Linear(hidden_size, message_size),
        )
        self.gru_node = nn.GRUCell(message_size, node_embed_size)
        self.gru_edge = nn.GRUCell(message_size, node_embed_size)

        self.device = device
        self.to(device)

    def forward(self, pos: torch.Tensor, classes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # pos.shape = (n_node, 3)
        # classes.shape = (n_node, n_classes)
        # edges.shape = (2, n_edges)

        # Initialize node features based on the node classes
        node_features = self.in_linear(classes)  # node_features.shape = (n_node, node_embed_size)

        # Symmetrise directional edge connections
        edges_sym = torch.cat([edges, edges[[1, 0]]], dim=1)

        # Compute vectors between nodes connected by edges
        src_pos = pos.index_select(0, edges_sym[0])
        dst_pos = pos.index_select(0, edges_sym[1])
        d_pos = dst_pos - src_pos

        # Initialize edge features to the average of the nodes they are connecting
        src_features = node_features.index_select(0, edges[0])
        dst_features = node_features.index_select(0, edges[1])
        edge_features = (src_features + dst_features) / 2

        for _ in range(self.iters):

            # Gather start and end nodes of edges
            src_features = node_features.index_select(0, edges_sym[0])
            dst_features = node_features.index_select(0, edges_sym[1])
            inputs = torch.cat([src_features, dst_features, d_pos], dim=1)

            # Calculate messages for all edges and add them to start nodes
            messages = self.msg_net(inputs)
            a = torch.zeros(node_features.size(0), self.message_size, device=self.device, dtype=messages.dtype)
            a.index_add_(0, edges_sym[0], messages)

            # Update node features
            node_features = self.gru_node(a, node_features)

            # Update edge features
            n_edge = edges.shape[1]
            b = (messages[:n_edge] + messages[n_edge:]) / 2  # Average over two directions
            edge_features = self.gru_edge(b, edge_features)

        return node_features


def get_edges(xyzs: list[torch.Tensor], Zs: list[torch.Tensor], tolerance: int = 0.2):
    edges = []
    for xyz, e in zip(xyzs, Zs):
        edge_ind = []
        for i in range(len(xyz)):
            for j in range(len(xyz)):
                if j <= i:
                    continue
                r = np.linalg.norm(xyz[i] - xyz[j])
                elems = sorted([e[i], e[j]])
                bond_length = BOND_LENGTHS[int(elems[0])][int(elems[1])]
                if r < (1 + tolerance) * bond_length:
                    edge_ind.append((i, j))
        edges.append(edge_ind)
    return edges


def make_graph(xyzs: list[torch.Tensor], Zs: list[torch.Tensor]):

    batch_nodes = [len(xyz) for xyz in xyzs]

    pos = torch.cat(xyzs, dim=0)

    edges = get_edges(xyzs, Zs)
    node_count = 0
    edges_combined = []
    for es, n_nodes in zip(edges, batch_nodes):
        es = torch.tensor(es).T + node_count
        edges_combined.append(es)
        node_count += n_nodes
    edges = torch.cat(edges_combined, dim=1)

    class_indices = []
    for elems in Zs:
        class_indices += [CLASSES.index(z) for z in elems]
    class_indices = torch.tensor(class_indices)
    classes = nn.functional.one_hot(class_indices, num_classes=len(CLASSES)).float()

    return pos, edges, classes, batch_nodes


def collate_graphs(samples: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor | list[torch.Tensor]]:

    # Convert to tensors
    batch = {k: [] for k in samples[0].keys()}
    for sample in samples:
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                batch[k].append(torch.from_numpy(v).float())
            else:
                batch[k].append(torch.tensor(v).float())

    # Pad densities with zeros in order to make them the same size
    max_shape, _ = torch.stack([torch.tensor(d.shape) for d in batch["density"]], dim=0).max(dim=0)
    padded_densities = []
    padded_lattices = []
    padded_origins = []
    for i_batch in range(len(samples)):
        d = batch["density"][i_batch]
        l = batch["lattice"][i_batch].diag()
        o = batch["origin"][i_batch]
        pad_total = max_shape - torch.tensor(d.shape)
        pad_start = torch.ceil(pad_total / 2).long()
        pad_end = torch.trunc(pad_total / 2).long()
        pad = (pad_start[2], pad_end[2], pad_start[1], pad_end[1], pad_start[0], pad_end[0])
        padded_densities.append(torch.nn.functional.pad(d, pad, mode="constant", value=0))
        grid_step = l / torch.tensor(d.shape)
        padded_lattices.append(torch.diag(l + grid_step * pad_total))
        padded_origins.append(o - grid_step * pad_start)
    batch["density"] = padded_densities
    batch["lattice"] = padded_lattices
    batch["origin"] = padded_origins

    # Stack samples into single batch tensors
    for k, v in batch.items():
        if k not in ["xyzs", "Zs"]:  # Molecules have variable number of atoms, so cannot stack them
            batch[k] = torch.stack(v, dim=0)

    # Combine all molecules to one graph
    pos, edges, classes, batch_nodes = make_graph(batch["xyzs"], batch["Zs"])
    batch["pos"] = pos
    batch["edges"] = edges
    batch["classes"] = classes
    batch["batch_nodes"] = batch_nodes

    return batch
