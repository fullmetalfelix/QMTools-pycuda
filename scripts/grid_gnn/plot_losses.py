
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __name__ == "__main__":

    data_train = np.loadtxt(f"loss_log_train.csv", delimiter=",")
    data_val = np.loadtxt(f"loss_log_val.csv", delimiter=",")

    loss_labels = ["Total loss", "MSE loss", "Gradient MSE loss", "MAE", "Gradient MAE"]

    n_batch_train = data_train[:, 0]
    loss_train = data_train[:, 1:]

    n_batch_val = data_val[:, 0]
    n_epoch_val = data_val[:, 1]
    loss_val = data_val[:, 2:]

    n_losses = loss_val.shape[1]
    assert n_losses <= 6

    if n_losses > 3:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={"hspace": 0.3, "wspace": 0.25, "left": 0.02, "left": 0.05, "right": 0.98, "bottom": 0.06, "top": 0.94})
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.25, "left": 0.05, "right": 0.98})

    for i in range(n_losses):
        if i < loss_train.shape[1]:
            axes[i].semilogy(n_batch_train, loss_train[:, i], "C0", label="Train")
        axes[i].semilogy(n_batch_val, loss_val[:, i], "C1", label="Validation")
        axes[i].set_xlabel("Batch")
        axes[i].set_ylabel(loss_labels[i])
        axes[i].legend()

        batches_per_epoch = (n_batch_val[-1] - n_batch_val[0]) / (n_epoch_val[-1] - n_epoch_val[0])
        n_epoch_min = axes[i].get_xlim()[0] / batches_per_epoch
        n_epoch_max = axes[i].get_xlim()[1] / batches_per_epoch

        ax_top = plt.twiny(ax=axes[i])
        ax_top.set_xlim(n_epoch_min, n_epoch_max)
        ax_top.set_xlabel("Epoch")
        ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))

    for i in range(n_losses, len(axes)):
        axes[i].axis("off")

    plt.savefig(f"loss.png")
    plt.show()