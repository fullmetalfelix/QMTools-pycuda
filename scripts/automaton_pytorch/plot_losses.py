
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = np.loadtxt("loss_log.csv", delimiter=",")

    loss_curves = []
    cur_data = []
    cur_ind = 0
    for row in data:
        ind = row[0]
        if cur_ind != ind:
            loss_curves.append(cur_data)
            cur_data = []
            cur_ind = ind
        cur_data.append(row[2])
    loss_curves.append(cur_data)

    loss_curves = np.array(loss_curves)
    print(loss_curves.shape)

    for i_batch, losses in enumerate(loss_curves[:-1]):
        plt.semilogy(losses, label=f"Batch {i_batch}")
    plt.semilogy(loss_curves[-1], label=f"Test")
    plt.xlabel("Iteration")
    plt.ylabel("Density MSE")
    plt.legend()
    plt.show()