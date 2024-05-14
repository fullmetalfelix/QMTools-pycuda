#!/usr/bin/env python3

import re
from pathlib import Path
import sys

import numpy as np
import pycuda.driver as cuda

from qmtools import BasisSet, Grid, Molecule, QMTools, ANG2BOR

BASIS_SET = BasisSet("../data/cc-pvdz.bin")
GRID_STEP = 0.1  # Ångströms


def get_sample(mol_dir):

    # Load molecule
    mol = Molecule(mol_dir / "GEOM-B3LYP.xyz", mol_dir / "D-CCSD.npy", BASIS_SET)

    # Generate the electron density grid and the nuclear potential grid
    gridTemplate = Grid.DensityGrid(mol, GRID_STEP, 3.0, multiple=32)
    density_grid = QMTools.Compute_density(gridTemplate, mol, copyBack=False)

    # Transfer to the host
    density = np.zeros(density_grid.shape[1:], order="F", dtype=np.float32)
    cuda.memcpy_dtoh(density, density_grid.d_qube)
    print(density.shape)

    lattice = GRID_STEP * np.diag(density_grid.shape[1:])
    origin = np.array([density_grid.origin['x'], density_grid.origin['y'], density_grid.origin['z']]) / ANG2BOR

    return mol, density, lattice, origin


if __name__ == "__main__":

    out_dir = Path("/scratch/work/oinonen1/density_db")
    ccsd_dir = Path("/scratch/work/oinonen1/CCSD-CID")

    if len(sys.argv) < 3:
        print("Not enough arguments")
        sys.exit(1)

    # Division over multiple processes
    n_proc = int(sys.argv[1])
    i_proc = int(sys.argv[2])

    out_dir.mkdir(exist_ok=True)

    mol_dirs = list(ccsd_dir.glob("molecule_*_0"))
    mol_dirs = sorted(mol_dirs)
    mol_dirs = mol_dirs[i_proc::n_proc]

    n_mols = len(mol_dirs)
    print("Total number of molecules:", n_mols)

    for i, mol_dir in enumerate(mol_dirs):

        cid = int(re.findall("_(.*?)_", mol_dir.name)[0])
        out_file_path = out_dir / f"{cid}.npz"
        if out_file_path.exists():
            print(f"CID {cid} already done.")
            continue

        print(f"Molecule {i+1} / {n_mols}, CID: {cid}")

        try:

            mol, density, lattice, origin = get_sample(mol_dir)
            xyz = mol.coords / ANG2BOR
            Z = mol.types
            print(lattice, origin, density.dtype)

            np.savez(out_file_path, xyz=xyz, Z=Z, data=density, origin=origin, lattice=lattice)

        except Exception as e:
            print(f"Ran into an error:\n{e}")
