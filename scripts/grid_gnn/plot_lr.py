#!/usr/bin/env python3


from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __name__ == "__main__":

    data = np.loadtxt(f"loss_log_lr.csv", delimiter=",")

    lr = data[:, 1]
    loss = data[:, 2:]

    n_losses = loss.shape[1]
    fig, axes = plt.subplots(1, n_losses, figsize=(2 + 6 * n_losses, 5))
    for i in range(n_losses):
        axes[i].loglog(lr, loss[:, i])
        axes[i].set_xlabel("Batch")
        axes[i].set_ylabel("Loss")

    plt.savefig(f"lr_loss.png")
    plt.show()
