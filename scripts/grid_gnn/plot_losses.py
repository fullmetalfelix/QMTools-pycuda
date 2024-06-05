
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

if __name__ == "__main__":

    data_train = np.loadtxt(f"loss_log_train.csv", delimiter=",")
    data_val = np.loadtxt(f"loss_log_val.csv", delimiter=",")

    n_batch_train = data_train[:, 0]
    loss_train = data_train[:, 1]

    n_batch_val = data_val[:, 0]    
    n_epoch_val = data_val[:, 1]    
    loss_val = data_val[:, 2]    

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax1.semilogy(n_batch_train, loss_train)
    ax1.semilogy(n_batch_val, loss_val)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.legend(["Train", "Validation"])

    batches_per_epoch = (n_batch_val[-1] - n_batch_val[0]) / (n_epoch_val[-1] - n_epoch_val[0])
    n_epoch_min = ax1.get_xlim()[0] / batches_per_epoch
    n_epoch_max = ax1.get_xlim()[1] / batches_per_epoch

    ax_top = plt.twiny(ax=ax1)
    ax_top.set_xlim(n_epoch_min, n_epoch_max)
    ax_top.set_xlabel("Epoch")
    ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(f"loss.png")
    plt.show()