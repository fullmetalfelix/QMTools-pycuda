
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = np.loadtxt("loss_log_vae.csv", delimiter=",")

    n_batch = data[:, 0]
    loss_total = data[:, 1]
    loss_rec = data[:, 2]
    loss_kl = data[:, 3]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    ax1.semilogy(n_batch, loss_total)
    ax2.semilogy(n_batch, loss_rec)
    ax3.semilogy(n_batch, loss_kl)

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Batch")
    ax1.set_ylabel("Total loss")
    ax2.set_ylabel("Reconstruction loss")
    ax3.set_ylabel("KL loss")
    
    plt.savefig("loss_vae.png")
    plt.show()