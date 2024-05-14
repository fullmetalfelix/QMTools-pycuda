import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io.xsf import write_xsf
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from qmtools.diffusion import DensityDiffusionVAE, VAELoss


def save_to_xsf(file_path: Path, sample: dict[str, np.ndarray]):
    atoms = Atoms(numbers=sample["Zs"][0], positions=sample["xyzs"][0] - sample["origin"][0], cell=sample["lattice"][0], pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=sample["density"][0])


class DensityDataset(Dataset):

    def __init__(self, data_dir: Path, shuffle=False):
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
    density_shapes = np.stack([d.shape for d in batch["density"]], axis=0, dtype=np.int32)
    max_shape = density_shapes.max(axis=0)
    padded_densities = []
    for d in batch["density"]:
        pad_total = max_shape - np.array(d.shape, dtype=np.int32)
        pad_left = np.ceil(pad_total / 2).astype(np.int32)
        pad_right = np.trunc(pad_total / 2).astype(np.int32)
        pad = (pad_left[2], pad_right[2], pad_left[1], pad_right[1], pad_left[0], pad_right[0])
        d_padded = torch.nn.functional.pad(d, pad, "constant", 0)
        padded_densities.append(d_padded)
    batch["density"] = padded_densities

    # Stack samples into single batch tensors
    for k, v in batch.items():
        if k not in ["xyzs", "Zs"]:  # Molecules have variable number of atoms, so cannot stack them
            batch[k] = torch.stack(v, dim=0)

    return batch


def make_dataloader(data_dir: Path, batch_size: int):
    dataset = DensityDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, collate_fn=collate_samples)
    return dataloader, dataset


if __name__ == "__main__":

    device = "cuda"
    n_epoch = 1
    batch_size = 4
    data_dir = Path("/scratch/work/oinonen1/density_db")
    loss_log_path = Path("loss_log_vae.csv")
    checkpoint_dir = Path("checkpoints_vae")
    densities_dir = Path("densities_vae")

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    checkpoint_dir.mkdir(exist_ok=True)
    densities_dir.mkdir(exist_ok=True)
    with open(loss_log_path, "w"):
        pass

    model = DensityDiffusionVAE(device=device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = VAELoss(kl_weight=1e-8)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model total parameters:", num_params)

    try:
        # Train
        model.train()
        losses = []
        i_batch = 0
        for i_epoch in range(n_epoch):

            dataloader, dataset = make_dataloader(data_dir, batch_size)
            print(f"Epoch {i_epoch}. {len(dataset)} samples.")

            for sample in dataloader:

                q_ref = sample["density"].to(device)

                # Forward
                pred = model(q_ref, out_relu=False)
                loss = criterion(pred, q_ref)

                # Backward
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                loss = [l.item() for l in loss]
                losses.append(loss)

                if i_batch % 10 == 0:
                    print(f"{i_batch}: {np.mean(losses, axis=0)}")

                if i_batch % 100 == 0:

                    # Save density
                    sample_pred = deepcopy(sample)
                    sample_pred["density"] = pred[0].detach().cpu().numpy()
                    save_to_xsf(densities_dir / f"density_{i_batch}_ref.xsf", sample)
                    save_to_xsf(densities_dir / f"density_{i_batch}_pred.xsf", sample_pred)

                    # Save model
                    torch.save(model.state_dict(), checkpoint_dir / f"weights_{i_batch}.pth")

                    # Save loss to file
                    with open(loss_log_path, "a") as f:
                        loss = np.mean(losses, axis=0)
                        f.write(f"{i_batch},{loss[0]},{loss[1]},{loss[2]}\n")

                    losses = []

                i_batch += 1

    except KeyboardInterrupt:
        pass

    # Do a test run
    model.eval()
    sample = collate_samples([dataset[0]])
    with torch.no_grad():
        q_ref = sample["density"].to(device)
        pred = model(q_ref, out_relu=False)
        loss = criterion(pred, q_ref)

    sample_pred = deepcopy(sample)
    sample_pred["density"] = pred[0].detach().cpu().numpy()
    save_to_xsf(densities_dir / f"density_ref.xsf", sample)
    save_to_xsf(densities_dir / f"density_test.xsf", sample)

    with open(loss_log_path, "a") as f:
        loss = [l.item() for l in loss]
        f.write(f"{i_batch+100},{loss[0]},{loss[1]},{loss[2]}\n")
