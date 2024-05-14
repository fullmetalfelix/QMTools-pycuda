from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import torch
from ase import Atoms
from ase.io.xsf import write_xsf
from torch import nn
from torch.optim import Adam

from qmtools import ANG2BOR, BasisSet, Grid, Molecule, QMTools
from qmtools.automaton_pytorch import AutomatonPT


def save_to_xsf(file_path, arr, mol, grid_step, origin):
    coords = mol.coords / ANG2BOR - np.array([origin[c] for c in ["x", "y", "z"]]) / ANG2BOR
    cell = grid_step * np.diag(arr.shape)
    atoms = Atoms(numbers=mol.types, positions=coords, cell=cell, pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=arr)


def get_batch(grid_step=0.1):

    basisset = BasisSet("../../data/cc-pvdz.bin")
    qm = QMTools()

    folder = "../../data/molecule_29766_0/"
    mol = Molecule(folder + "GEOM-B3LYP.xyz", folder + "D-CCSD.npy", basisset)
    gridTemplate = Grid.DensityGrid(mol, grid_step, 3.0)

    # generate the electron grid from the DM and basis set
    q_ref_grid = QMTools.Compute_density(gridTemplate, mol, copyBack=False)
    gvne_grid = QMTools.Compute_VNe(gridTemplate, mol, adsorb=0.1, diff=0.01, tolerance=1.0e-9, copyBack=False)
    gqsd_grid = QMTools.Compute_qseed(gridTemplate, mol, copyBack=False)

    q_ref = np.zeros(q_ref_grid.shape[1:], order="F", dtype=np.float32)
    cuda.memcpy_dtoh(q_ref, q_ref_grid.d_qube)
    print(q_ref.shape)

    state = np.zeros(gvne_grid.shape[1:] + (2,), order="F", dtype=np.float32)
    cuda.memcpy_dtoh(state[..., 0], gvne_grid.d_qube)
    cuda.memcpy_dtoh(state[..., 1], gqsd_grid.d_qube)

    q_ref = torch.tensor(q_ref).unsqueeze(0)
    state = torch.tensor(state).unsqueeze(0)

    return q_ref, state, mol, q_ref_grid.origin


if __name__ == "__main__":

    device = "cuda"
    grid_step = 0.10
    n_batch = 20
    n_iter = 1000
    loss_log_path = Path("loss_log.csv")
    checkpoint_dir = Path("checkpoints")
    densities_dir = Path("densities")

    checkpoint_dir.mkdir(exist_ok=True)
    densities_dir.mkdir(exist_ok=True)
    with open(loss_log_path, "w"):
        pass

    # This needs to come before the Pytorch model is initialized. Somehow pycuda does not play well with Pytorch in the same process.
    q_ref, state, mol, origin = get_batch(grid_step)
    save_to_xsf(densities_dir / f"density_ref.xsf", q_ref[0], mol, grid_step, origin)
    q_ref = q_ref.to(device)
    state_init = state.to(device)

    model = AutomatonPT(n_layer=4, device=device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train
    for i_batch in range(n_batch):

        print(i_batch)
        f_loss_log = open(loss_log_path, "a")

        # Copy the initial state so that we start from the same place every time
        state = state_init.clone()

        for iter in range(n_iter):

            # Forward
            state = model(state)
            q_pred = state[..., 0]
            loss = criterion(q_pred, q_ref)
            print(iter, loss)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Need to detach so that on the following iteration the computational graph does not extend back to this iteration
            state = state.detach()

            # Save loss to file
            f_loss_log.write(f"{i_batch},{iter},{loss.item()}\n")

        f_loss_log.close()

        # Save current model weights
        torch.save(model.state_dict(), checkpoint_dir / f"weights_{i_batch}.pth")

        # Save final density
        q_pred = q_pred[0].detach().cpu().numpy()
        diff = q_pred - q_ref[0].cpu().numpy()
        save_to_xsf(densities_dir / f"density_{i_batch}.xsf", q_pred, mol, grid_step, origin)
        save_to_xsf(densities_dir / f"diff_{i_batch}.xsf", diff, mol, grid_step, origin)

    # Do a test run where the weights are fixed
    state = state_init.clone()
    for iter in range(n_iter):

        # Forward
        with torch.no_grad():
            state = model(state)
            q_pred = state[..., 0]
            loss = criterion(q_pred, q_ref)

        # Save loss to file
        with open(loss_log_path, "a") as f:
            f.write(f"{n_batch+1},{iter},{loss.item()}\n")
