#!/usr/bin/env python3

from pathlib import Path
from ase.io.xsf import read_xsf
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    i_batch = 8
    densities_dir = Path("densities")

    with open(densities_dir / "density_ref.xsf", "r") as f:
        q_ref, _ = read_xsf(f, read_data=True)
    with open(densities_dir / f"density_{i_batch}.xsf", "r") as f:
        q_pred, _ = read_xsf(f, read_data=True)

    q_ref = q_ref.flatten()
    q_pred = q_pred.flatten()

    q_min = min(q_ref.min(), q_ref.min())
    q_max = max(q_ref.max(), q_ref.max())
    print(q_min, q_max)

    plt.plot([q_min, q_max], [q_min, q_max])
    plt.scatter(q_ref, q_pred, s=0.5, c="r", marker="o")
    plt.xlabel("True density")
    plt.ylabel("Predicted density")
    plt.axis("equal")
    plt.show()
