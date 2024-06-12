
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __name__ == "__main__":

    data_train = np.loadtxt(f"loss_log_train.csv", delimiter=",")
    data_val = np.loadtxt(f"loss_log_val.csv", delimiter=",")

    loss_labels = ["Total loss", "MSE loss", "Gradient MSE loss"]

    n_batch_train = data_train[:, 0]
    loss_train = data_train[:, 1:]

    n_batch_val = data_val[:, 0]
    n_epoch_val = data_val[:, 1]
    loss_val = data_val[:, 2:]

    assert loss_train.shape[1] == loss_train.shape[1]
    n_losses = loss_train.shape[1]

    fig, axes = plt.subplots(1, n_losses, figsize=(2 + 6 * n_losses, 5))
    for i in range(n_losses):
        axes[i].semilogy(n_batch_train, loss_train[:, i])
        axes[i].semilogy(n_batch_val, loss_val[:, i])
        axes[i].set_xlabel("Batch")
        axes[i].set_ylabel(loss_labels[i])
        axes[i].legend(["Train", "Validation"])

        batches_per_epoch = (n_batch_val[-1] - n_batch_val[0]) / (n_epoch_val[-1] - n_epoch_val[0])
        n_epoch_min = axes[i].get_xlim()[0] / batches_per_epoch
        n_epoch_max = axes[i].get_xlim()[1] / batches_per_epoch

        ax_top = plt.twiny(ax=axes[i])
        ax_top.set_xlim(n_epoch_min, n_epoch_max)
        ax_top.set_xlabel("Epoch")
        ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(f"loss.png")
    plt.show()