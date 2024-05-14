import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ase import Atoms
from ase.io.xsf import write_xsf
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from qmtools.diffusion import MPNNEncoder, DensitySRDecoder

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


def save_to_xsf(file_path: Path, sample: dict[str, np.ndarray]):
    atoms = Atoms(numbers=sample["Zs"][0], positions=sample["xyzs"][0] - sample["origin"][0], cell=sample["lattice"][0], pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=sample["density"][0])


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


def collate_samples(samples: list[dict[str, np.ndarray | int]]):

    # Convert to tensors
    batch = {k: [] for k in samples[0].keys()}
    for sample in samples:
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                batch[k].append(torch.from_numpy(v))
            else:
                batch[k].append(torch.tensor(v))

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

    # Make input by down-sampling the reference density
    batch["input"] = (4**3) * nn.functional.avg_pool3d(batch["density"], kernel_size=4, stride=4)

    # Combine all molecules to one graph
    pos, edges, classes, batch_nodes = make_graph(batch["xyzs"], batch["Zs"])
    batch["pos"] = pos
    batch["edges"] = edges
    batch["classes"] = classes
    batch["batch_nodes"] = batch_nodes

    return batch


def make_dataloader(data_dir: Path, batch_size: int):
    dataset = DensityDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, collate_fn=collate_samples)
    return dataloader, dataset


class DensityDataset(Dataset):

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sample_paths = sorted(list(data_dir.glob("*.npz")))

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample_path = self.sample_paths[index]
        sample_npz = np.load(sample_path)
        sample = {
            "density": sample_npz["data"],
            "xyzs": sample_npz["xyz"],
            "Zs": sample_npz["Z"],
            "lattice": sample_npz["lattice"],
            "origin": sample_npz["origin"],
            "index": index,
        }
        sample_npz.close()
        return sample

    def shuffle(self):
        random.shuffle(self.sample_paths)


if __name__ == "__main__":

    device = "cuda"
    n_epoch = 1
    batch_size = 8
    mol_embed_size = 32
    data_dir = Path("/scratch/work/oinonen1/density_db")
    loss_log_path = Path("loss_log_sr.csv")
    checkpoint_dir = Path("checkpoints_sr")
    densities_dir = Path("densities_sr")

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    checkpoint_dir.mkdir(exist_ok=True)
    densities_dir.mkdir(exist_ok=True)
    with open(loss_log_path, "w"):
        pass

    mpnn_encoder = MPNNEncoder(n_class=len(CLASSES), node_embed_size=mol_embed_size, hidden_size=32, message_size=32, device=device)
    sr_decoder = DensitySRDecoder(mol_embed_size=mol_embed_size, device=device)
    optimizer = Adam(list(mpnn_encoder.parameters()) + list(sr_decoder.parameters()), lr=1e-4)
    criterion = nn.MSELoss(reduction="mean")

    print("Encoder total parameters:", sum(p.numel() for p in mpnn_encoder.parameters() if p.requires_grad))
    print("Decoder total parameters:", sum(p.numel() for p in sr_decoder.parameters() if p.requires_grad))

    try:
        # Train
        mpnn_encoder.train()
        sr_decoder.train()
        losses = []
        i_batch = 0
        for i_epoch in range(n_epoch):

            dataloader, dataset = make_dataloader(data_dir, batch_size)
            print(f"Epoch {i_epoch}. {len(dataset)} samples.")

            for batch in dataloader:

                q_input = batch["input"].to(device)
                q_ref = batch["density"].to(device)
                pos = batch["pos"].to(device)
                classes = batch["classes"].to(device)
                edges = batch["edges"].to(device)
                batch_nodes = batch["batch_nodes"]

                # Forward
                mol_embed = mpnn_encoder(pos, classes, edges, batch_nodes)
                q_pred = sr_decoder(q_input, mol_embed, batch_nodes)
                loss = criterion(q_pred, q_ref)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.item()
                losses.append(loss)

                if i_batch % 1 == 0:
                    print(f"{i_batch}: {np.mean(losses)}")

                if i_batch % 100 == 0:

                    # Save density
                    batch_pred = deepcopy(batch)
                    batch_pred["density"] = q_pred.detach().cpu().numpy()
                    save_to_xsf(densities_dir / f"density_{i_batch}_ref.xsf", batch)
                    save_to_xsf(densities_dir / f"density_{i_batch}_pred.xsf", batch_pred)

                    # Save model
                    torch.save(
                        {
                            "mpnn_encoder": mpnn_encoder.state_dict(),
                            "sr_decoder": sr_decoder.state_dict(),
                        },
                        checkpoint_dir / f"weights_{i_batch}.pth",
                    )

                    # Save loss to file
                    with open(loss_log_path, "a") as f:
                        loss = np.mean(losses)
                        f.write(f"{i_batch},{loss}\n")

                    losses = []

                i_batch += 1

    except KeyboardInterrupt:
        pass

    # Do a test run
    mpnn_encoder.eval()
    sr_decoder.eval()
    batch = collate_samples([dataset[0]])
    with torch.no_grad():
        q_input = batch["input"].to(device)
        q_ref = batch["density"].to(device)
        pos = batch["pos"].to(device)
        classes = batch["classes"].to(device)
        edges = batch["edges"].to(device)
        batch_nodes = batch["batch_nodes"]
        mol_embed = mpnn_encoder(pos, classes, edges, batch_nodes)
        q_pred = sr_decoder(q_input, mol_embed, batch_nodes)
        loss = criterion(q_pred, q_ref)

    batch_pred = deepcopy(batch)
    batch_pred["density"] = q_pred.detach().cpu().numpy()
    save_to_xsf(densities_dir / f"density_test_ref.xsf", batch)
    save_to_xsf(densities_dir / f"density_test_pred.xsf", batch_pred)

    with open(loss_log_path, "a") as f:
        loss = loss.item()
        f.write(f"{i_batch+100},{loss}\n")
