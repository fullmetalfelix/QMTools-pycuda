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

from qmtools.pt.gnn import MPNNEncoder, collate_graphs, CLASSES
from qmtools.pt.sr import AtomGrid, DensitySRDecoder


def save_to_xsf(file_path: Path, sample: dict[str, np.ndarray]):
    atoms = Atoms(numbers=sample["Zs"][0], positions=sample["xyzs"][0] - sample["origin"][0], cell=sample["lattice"][0], pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=sample["density"][0])


def make_dataloader(data_dir: Path, batch_size: int):
    dataset = DensityDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, collate_fn=collate_graphs)
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
                classes = batch["classes"].to(device)
                edges = batch["edges"].to(device)
                atom_grid = AtomGrid(batch["pos"], batch["origin"], batch["lattice"], device=device)
                batch_nodes = batch["batch_nodes"]

                # Forward
                mol_embed = mpnn_encoder(atom_grid.pos, classes, edges)
                mol_embed, atom_grid = mpnn_encoder.split_graph(mol_embed, atom_grid, batch_nodes)
                q_pred = sr_decoder(q_input, mol_embed, atom_grid, batch_nodes)
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
    batch = collate_graphs([dataset[0]])
    with torch.no_grad():

        q_input = batch["input"].to(device)
        q_ref = batch["density"].to(device)
        classes = batch["classes"].to(device)
        edges = batch["edges"].to(device)
        atom_grid = AtomGrid(batch["pos"], batch["origin"], batch["lattice"], device=device)
        batch_nodes = batch["batch_nodes"]

        mol_embed = mpnn_encoder(atom_grid.pos, classes, edges)
        mol_embed, atom_grid = mpnn_encoder.split_graph(mol_embed, atom_grid, batch_nodes)
        q_pred = sr_decoder(q_input, mol_embed, atom_grid, batch_nodes)
        loss = criterion(q_pred, q_ref)

    batch_pred = deepcopy(batch)
    batch_pred["density"] = q_pred.detach().cpu().numpy()
    save_to_xsf(densities_dir / f"density_test_ref.xsf", batch)
    save_to_xsf(densities_dir / f"density_test_pred.xsf", batch_pred)

    with open(loss_log_path, "a") as f:
        loss = loss.item()
        f.write(f"{i_batch+100},{loss}\n")
