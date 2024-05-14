
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = np.loadtxt(f"loss_log_sr.csv", delimiter=",")

    n_batch = data[:, 0]
    loss = data[:, 1]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax1.semilogy(n_batch, loss)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Total loss")
    
    plt.savefig(f"loss_sr.png")
    plt.show()