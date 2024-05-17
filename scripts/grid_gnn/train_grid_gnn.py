import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ase import Atoms
from ase.io.xsf import write_xsf
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.append("../..")
from qmtools.pt.gnn import MPNNEncoder, collate_graphs, CLASSES
from qmtools.pt.grid import AtomGrid, DensityGridNN


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
    n_epoch = 5
    batch_size = 2
    # data_dir = Path("/scratch/work/oinonen1/density_db")
    data_dir = Path("/mnt/triton/density_db")
    loss_log_path = Path("loss_log.csv")
    checkpoint_dir = Path("checkpoints")
    densities_dir = Path("densities")

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    checkpoint_dir.mkdir(exist_ok=True)
    densities_dir.mkdir(exist_ok=True)
    with open(loss_log_path, "w"):
        pass

    mpnn_encoder = MPNNEncoder(n_class=len(CLASSES), iters=12, node_embed_size=128, hidden_size=128, message_size=128, device=device)
    model = DensityGridNN(mpnn_encoder, device=device)
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    criterion = nn.MSELoss(reduction="mean")

    print("Encoder parameters:", sum(p.numel() for p in mpnn_encoder.parameters() if p.requires_grad))
    print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    try:
        # Train
        model.train()
        losses = []
        i_batch = 0
        for i_epoch in range(n_epoch):

            dataloader, dataset = make_dataloader(data_dir, batch_size)
            print(f"Epoch {i_epoch}. {len(dataset)} samples.")

            for batch in dataloader:

                q_ref = batch["density"].to(device)
                classes = batch["classes"].to(device)
                edges = batch["edges"].to(device)
                atom_grid = AtomGrid(batch["pos"], batch["origin"], batch["lattice"], q_ref.shape[1:], device=device)
                batch_nodes = batch["batch_nodes"]

                # Forward
                q_pred, c = model(atom_grid, classes, edges, batch_nodes)
                loss = criterion(q_pred, q_ref)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss = loss.item()
                losses.append(loss)

                if i_batch % 1 == 0:
                    print(f"{i_batch}: {np.mean(losses)}")

                if i_batch % 100 == 0:

                    print("Current learning rate:", scheduler.get_last_lr())

                    # Save density
                    batch_pred = deepcopy(batch)
                    batch_pred["density"] = q_pred.detach().cpu().numpy()
                    save_to_xsf(densities_dir / f"density_{i_batch}_ref.xsf", batch)
                    save_to_xsf(densities_dir / f"density_{i_batch}_pred.xsf", batch_pred)

                    c = c.detach().cpu().numpy()
                    for k in range(8):
                        batch_c = deepcopy(batch)
                        batch_c["density"] = c[:, :, :, :, k]
                        save_to_xsf(densities_dir / f"density_{i_batch}_c{k}.xsf", batch_c)
                        

                    # Save model
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
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
    model.eval()
    batch = collate_graphs([dataset[0]])
    with torch.no_grad():

        q_ref = batch["density"].to(device)
        classes = batch["classes"].to(device)
        edges = batch["edges"].to(device)
        atom_grid = AtomGrid(batch["pos"], batch["origin"], batch["lattice"], q_ref.shape[1:], device=device)
        batch_nodes = batch["batch_nodes"]

        q_pred = model(atom_grid, classes, edges, batch_nodes)
        loss = criterion(q_pred, q_ref)

    batch_pred = deepcopy(batch)
    batch_pred["density"] = q_pred.detach().cpu().numpy()
    save_to_xsf(densities_dir / f"density_test_ref.xsf", batch)
    save_to_xsf(densities_dir / f"density_test_pred.xsf", batch_pred)

    with open(loss_log_path, "a") as f:
        loss = loss.item()
        f.write(f"{i_batch+100},{loss}\n")
