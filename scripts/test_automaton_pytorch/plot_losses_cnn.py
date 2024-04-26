
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = np.loadtxt("loss_log_cnn.csv", delimiter=",")
    batches = data[:, 0]
    losses = data[:, 1]

    plt.semilogy(batches, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Density MSE")
    plt.savefig("losses_cnn.png")
    plt.show()