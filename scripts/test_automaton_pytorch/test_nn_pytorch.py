
import torch
from qmtools.automaton_pytorch import AutomatonPT

if __name__ == "__main__":

    model = AutomatonPT(n_layer=4)

    n_batch = 2
    n_voxel = 100
    x = torch.rand(n_batch, n_voxel, 26, 4)
    t = model(x)
    print(t.shape)