import math
import os
import pickle
import random
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from ase import Atoms
from ase.io.xsf import write_xsf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset

sys.path.append("../..")
from qmtools.pt.gnn import CLASSES, MPNNEncoder, collate_graphs
from qmtools.pt.grid import AtomGrid, DensityGridNN, GridLoss


def save_to_xsf(file_path: Path, sample: dict[str, np.ndarray]):
    atoms = Atoms(numbers=sample["Zs"][0], positions=sample["xyzs"][0] - sample["origin"][0], cell=sample["lattice"][0], pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=sample["density"][0])


def make_dataloader(data_dir: Path, sample_paths: list[int], batch_size: int, rank: int, world_size: int):
    dataset = DensityDataset(data_dir, sample_paths, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, collate_fn=collate_graphs)
    return dataloader, dataset


class DensityDataset(Dataset):

    def __init__(self, data_dir: Path, sample_paths: list[Path], rank: int, world_size: int):
        self.data_dir = data_dir
        # Flooring here can result in leaving samples unused, but gives consistent number of samples across ranks
        chunk_size = math.floor(len(sample_paths) / world_size)
        sample_paths = sample_paths[rank * chunk_size : (rank + 1) * chunk_size]
        self.sample_paths = [data_dir / p for p in sample_paths]

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

def lr_schedule(i_batch, lr_init=1e-10, T_warm=1000, T_decay=10000):
    if i_batch <= T_warm:
        lr = lr_init + (1 - lr_init) * (i_batch / T_warm)
    else:
        lr = 1 / (1 + (i_batch - T_warm) / T_decay)
    return lr

def run(local_rank, global_rank, world_size):

    print(f"Starting on global rank {global_rank}, local rank {local_rank}\n", flush=True)

    # Initialize the distributed environment.
    dist.init_process_group("nccl")

    device = local_rank
    n_epoch = 50
    batch_size = 4
    use_amp = True
    data_dir = Path("/scratch/work/oinonen1/density_db")
    # data_dir = Path("/mnt/triton/density_db")
    sample_dict_path = Path("../sample_dict.pickle")
    loss_log_path_train = Path("loss_log_train.csv")
    loss_log_path_val = Path("loss_log_val.csv")
    checkpoint_dir = Path("checkpoints")
    densities_dir = Path("densities")

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    checkpoint_dir.mkdir(exist_ok=True)
    densities_dir.mkdir(exist_ok=True)

    mpnn_encoder = MPNNEncoder(
        n_class=len(CLASSES),
        iters=12,
        node_embed_size=128,
        hidden_size=128,
        message_size=128,
        device=device,
    )
    model = DensityGridNN(
        mpnn_encoder,
        proj_channels=[64, 16, 4],
        cnn_channels=[128, 64, 32],
        lorentz_type=2,
        scale_init_bounds=(0.5, 1.5),
        device=device,
    )
    optimizer = Adam(model.parameters(), lr=4e-3)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda nb: lr_schedule(nb, T_warm=4000, T_decay=10000))
    criterion = GridLoss(grad_factor=1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_nums = sorted([int(re.search("[0-9]+", p.name).group(0)) for p in checkpoint_dir.glob("weights_*.pth")])
    if len(checkpoint_nums) > 0:
        prev_epoch = checkpoint_nums[-1]
        init_epoch = prev_epoch + 1
        if global_rank == 0:
            print(f"Continuing from epoch {init_epoch}")
        checkpoint = torch.load(checkpoint_dir / f"weights_{prev_epoch}.pth", map_location={"cuda:0": f"cuda:{local_rank}"})
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        i_batch = checkpoint["i_batch"]
    else:
        init_epoch = 1
        i_batch = 0

    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)

    if global_rank == 0:
        print("Encoder parameters:", sum(p.numel() for p in mpnn_encoder.parameters() if p.requires_grad))
        print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    with open(sample_dict_path, "rb") as f:
        sample_dict = pickle.load(f)
        samples_train, samples_val = [sample_dict[s] for s in ["train", "val"]]
    dataloader_train, dataset_train = make_dataloader(data_dir, samples_train, batch_size, global_rank, world_size)
    dataloader_val, dataset_val = make_dataloader(data_dir, samples_val, batch_size, global_rank, world_size)
    print(f"Rank {global_rank}: ")
    print(f"{len(dataset_train)} train samples, {math.ceil(len(dataset_train) / batch_size)} batches in an epoch.")
    print(f"{len(dataset_val)} val samples, {math.ceil(len(dataset_val) / batch_size)} batches in an epoch.")

    losses = []
    for i_epoch in range(init_epoch, n_epoch + 1):

        if global_rank == 0:
            print(f"Epoch {i_epoch}")

        model.train()
        for batch in dataloader_train:

            q_ref = batch["density"].to(device)
            classes = batch["classes"].to(device)
            edges = batch["edges"].to(device)
            atom_grid = AtomGrid(batch["pos"], batch["origin"], batch["lattice"], q_ref.shape[1:], device=device)
            batch_nodes = batch["batch_nodes"]

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                q_pred = model(atom_grid, classes, edges, batch_nodes)
                loss = criterion(q_pred, q_ref)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss[0]).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss = [l.item() for l in loss]
            losses.append(loss)

            if global_rank == 0:
                print(f"{i_batch}: {np.mean(losses, axis=0)}")

            if i_batch % 1000 == 0:

                loss = np.mean(losses, axis=0)
                if world_size > 1:
                    # Take an average of the loss value across parallel ranks
                    loss = torch.tensor(loss).to(device)
                    dist.all_reduce(loss, dist.ReduceOp.SUM)
                    loss = loss.cpu().numpy()
                    loss /= world_size

                if global_rank == 0:

                    print("Current learning rate:", scheduler.get_last_lr())
                    print(list(model.module.scale))
                    if model.module.amplitude:
                        print(list(model.module.amplitude))

                    # Save loss to file
                    with open(loss_log_path_train, "a") as f:
                        loss_str = ",".join(str(l) for l in loss)
                        f.write(f"{i_batch},{loss_str}\n")

                losses = []

            i_batch += 1

        # Validation
        losses_val = []
        for i_val_batch, batch in enumerate(dataloader_val):

            q_ref = batch["density"].to(device)
            classes = batch["classes"].to(device)
            edges = batch["edges"].to(device)
            atom_grid = AtomGrid(batch["pos"], batch["origin"], batch["lattice"], q_ref.shape[1:], device=device)
            batch_nodes = batch["batch_nodes"]

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    q_pred = model(atom_grid, classes, edges, batch_nodes)
                    loss = criterion(q_pred, q_ref)

            loss = [l.item() for l in loss]
            losses_val.append(loss)

            if global_rank == 0:
                print(f"val {i_val_batch}: {np.mean(losses_val, axis=0)}")

        val_loss = np.mean(losses_val, axis=0)
        if world_size > 1:
            # Take an average of the loss value across parallel ranks
            val_loss = torch.tensor(val_loss).to(device)
            dist.all_reduce(val_loss, dist.ReduceOp.SUM)
            val_loss = val_loss.cpu().numpy()
            val_loss /= world_size

        if global_rank == 0:

            # Save validation loss to file
            with open(loss_log_path_val, "a") as f:
                loss_str = ",".join(str(l) for l in val_loss)
                f.write(f"{i_batch}, {i_epoch},{loss_str}\n")

            # Save density from last validation batch
            batch_pred = deepcopy(batch)
            batch_pred["density"] = q_pred.detach().cpu().numpy()
            save_to_xsf(densities_dir / f"density_{i_epoch}_ref.xsf", batch)
            save_to_xsf(densities_dir / f"density_{i_epoch}_pred.xsf", batch_pred)

            # Save model
            torch.save(
                {
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "i_batch": i_batch,
                },
                checkpoint_dir / f"weights_{i_epoch}.pth",
            )


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    run(local_rank, global_rank, world_size)
