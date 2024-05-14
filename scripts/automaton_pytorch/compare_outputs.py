import numpy as np
import pycuda.driver as cuda
import torch
from ase import Atoms
from ase.io.xsf import write_xsf
from pycuda.compiler import SourceModule
from testnn_simple import save_to_xsf
from torch import nn
from torch.optim import Adam

from qmtools import ANG2BOR, AutomatonNNSimple, BasisSet, Grid, Molecule, QMTools
from qmtools.pt.automaton import AutomatonPT
from qmtools.qmtools import CUDA_SRC_DIR


def save_to_xsf(file_path, arr, mol, grid_step, origin):
    coords = mol.coords / ANG2BOR - np.array([origin[c] for c in ["x", "y", "z"]]) / ANG2BOR
    cell = (grid_step / ANG2BOR) * np.diag(arr.shape)
    atoms = Atoms(numbers=mol.types, positions=coords, cell=cell, pbc=True)
    with open(file_path, "w") as f:
        write_xsf(f, [atoms], data=arr)


def automaton_compute_density(grid_step):

    basisset = BasisSet("../../data/cc-pvdz.bin")
    qm = QMTools()

    folder = "../../data/molecule_29766_0/"
    mol = Molecule(folder + "GEOM-B3LYP.xyz", folder + "D-CCSD.npy", basisset)
    gridTemplate = Grid.DensityGrid(mol, grid_step, 3.0)

    # generate the electron grid from the DM and basis set
    qref = QMTools.Compute_density(gridTemplate, mol, copyBack=True)
    np.save("pycuda_qref.npy", qref.qube)

    qref = Grid.emptyAs(gridTemplate)
    qref.LoadData("pycuda_qref.npy")

    gvne = QMTools.Compute_VNe(gridTemplate, mol, adsorb=0.1, diff=0.01, tolerance=1.0e-9, copyBack=True)
    gqsd = QMTools.Compute_qseed(gridTemplate, mol, copyBack=True)

    # the compute grid is allocated separately so we can reuse it
    cgrid = Grid.emptyAs(gridTemplate, nfields=4)

    # create an automaton and initialize the compute grid
    atm = AutomatonNNSimple()
    atm.Randomize(5.0, 4)
    atm.Initialize(cgrid, gqsd, gvne)

    kernel = atm.WriteCode()
    kernel = kernel.replace("PYCUDA_NX", str(cgrid.shape[1]))
    kernel = kernel.replace("PYCUDA_NY", str(cgrid.shape[2]))
    kernel = kernel.replace("PYCUDA_NZ", str(cgrid.shape[3]))
    kernel = kernel.replace("PYCUDA_NPTS", str(cgrid.npts))
    kernel = kernel.replace("PYCUDA_GRIDSTEP", str(cgrid.step))

    ptx = SourceModule(kernel, include_dirs=[str(CUDA_SRC_DIR)])
    cp, sb = ptx.get_global("cParams")
    params = np.asarray(atm.params).astype(np.float32)
    cuda.memcpy_htod(cp, params)  # copy constant param
    ptx = ptx.get_function("gpu_automaton_nn_simple_evolve")
    ptx.prepare([np.intp, np.intp])

    ogrid = Grid.emptyAs(cgrid)
    ptx.prepared_call(cgrid.GPUblocks, (8, 8, 8), cgrid.d_qube, ogrid.d_qube)

    arr_ogrid = np.zeros(ogrid.shape[1:] + (ogrid.shape[0],), order="F", dtype=np.float32)
    cuda.memcpy_dtoh(arr_ogrid, ogrid.d_qube)

    arr_cgrid = np.zeros(cgrid.shape[1:] + (cgrid.shape[0],), order="F", dtype=np.float32)
    cuda.memcpy_dtoh(arr_cgrid, cgrid.d_qube)

    return arr_cgrid[..., :2], arr_ogrid[..., :2], atm, mol, ogrid.origin


if __name__ == "__main__":

    device = "cuda"
    grid_step = 0.15

    np.random.seed(0)

    # Construct automaton and run one iteration
    arr_cgrid, output_atm, atm, mol, origin = automaton_compute_density(grid_step)

    # Construct pytorch model and copy parameters from the automaton
    model = AutomatonPT(n_layer=4, device=device)
    state_dict = {"extra_constants": torch.tensor(atm.params[:12])}
    ind = 12
    for layer in range(4):
        layer_ind = layer * 2
        # layer_ind = layer
        weight = []
        bias = []
        for i in range(16):
            weight.append(atm.params[ind : ind + 16])
            ind += 16
            bias.append(atm.params[ind])
            ind += 1
        weight = np.stack(weight, axis=0)
        bias = np.array(bias)
        state_dict[f"net.{layer_ind}.weight"] = torch.tensor(weight)
        state_dict[f"net.{layer_ind}.bias"] = torch.tensor(bias)
    state_dict["net.8.weight"] = torch.tensor(atm.params[ind: ind + 16]).reshape(1, 16)
    ind += 16
    state_dict["net.8.bias"] = torch.tensor(atm.params[ind : ind + 1])
    model.load_state_dict(state_dict)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    x = torch.tensor(arr_cgrid, device=device).unsqueeze(0)
    print(x.shape)
    output_pt = model(x).squeeze(0).detach().cpu().numpy()

    diff = output_atm[..., 0] - output_pt[..., 0]
    diff_rel = diff / output_atm[..., 0]
    diff_rel[np.isnan(diff_rel)] = 0
    print(np.allclose(output_pt, output_atm, atol=1e-6, rtol=1e-4))
    print(diff.min(), diff.max())
    print(diff_rel.min(), diff_rel.max())

    save_to_xsf("output_atm.xsf", output_atm[..., 0], mol, grid_step, origin)
    save_to_xsf("output_pt.xsf", output_pt[..., 0], mol, grid_step, origin)
    save_to_xsf("output_diff.xsf", diff, mol, grid_step, origin)
