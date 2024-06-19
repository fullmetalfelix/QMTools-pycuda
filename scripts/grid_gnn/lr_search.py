import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset

sys.path.append("../..")
from qmtools.pt.gnn import CLASSES, MPNNEncoder, collate_graphs
from qmtools.pt.grid import AtomGrid, DensityGridNN, GridLoss


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


def run(local_rank, global_rank, world_size):

    print(f"Starting on global rank {global_rank}, local rank {local_rank}\n", flush=True)

    # Initialize the distributed environment.
    dist.init_process_group("nccl")

    device = local_rank
    batch_size = 4
    use_amp = True
    data_dir = Path("/scratch/work/oinonen1/density_db")
    sample_dict_path = Path("./sample_dict.pickle")
    loss_log_path = Path("loss_log_lr.csv")

    if global_rank == 0 and loss_log_path.exists():
        loss_log_path.unlink()

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

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
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda nb: 1.05**nb)
    criterion = GridLoss(grad_factor=1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)

    if global_rank == 0:
        print("Encoder parameters:", sum(p.numel() for p in mpnn_encoder.parameters() if p.requires_grad))
        print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    with open(sample_dict_path, "rb") as f:
        sample_dict = pickle.load(f)
        samples_train = sample_dict["train"]
    dataloader_train, _ = make_dataloader(data_dir, samples_train, batch_size, global_rank, world_size)
    print(f"Rank {global_rank}: ")

    model.train()
    for i_batch, batch in enumerate(dataloader_train):

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

        loss = torch.tensor(loss, device=device)
        if world_size > 1:
            # Take an average of the loss value across parallel ranks
            dist.all_reduce(loss, dist.ReduceOp.SUM)
        loss = loss.cpu().numpy()
        loss /= world_size

        if global_rank == 0:

            lr = scheduler.get_last_lr()[0]
            print(f"Batch {i_batch}, learning rate: {lr}, loss: {loss}")

            # Save loss to file
            with open(loss_log_path, "a") as f:
                loss_str = ",".join(str(l) for l in loss)
                f.write(f"{i_batch},{lr},{loss_str}\n")

        if i_batch == 0:
            loss_init = loss
        elif loss[0] > loss_init[0]:
            break


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    run(local_rank, global_rank, world_size)
