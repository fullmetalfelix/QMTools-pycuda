import os
from ase import Atoms
from ase.io.xsf import write_xsf

from matplotlib import pyplot as plt
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from qmtools import ANG2BOR, AutomatonNNSimple, BasisSet, Grid, Molecule, QMTools

if __name__ == "__main__":

    basisset = BasisSet("cc-pvdz.bin")
    qm = QMTools()

    folder = "./molecule_29766_0/"
    mol = Molecule(folder + "GEOM-B3LYP.xyz", folder + "D-CCSD.npy", basisset)

    gridTemplate = Grid.DensityGrid(mol, 0.1, 3.0)

    # generate the electron grid from the DM and basis set
    # qref = QMTools.Compute_density(gridTemplate, mol, copyBack=True)
    # np.save("pycuda_qref.npy", qref.qube)

    qref = Grid.emptyAs(gridTemplate)
    qref.LoadData("pycuda_qref.npy")

    gvne = QMTools.Compute_VNe(gridTemplate, mol, adsorb=0.1, diff=0.01, tolerance=1.0e-9, copyBack=True)
    gqsd = QMTools.Compute_qseed(gridTemplate, mol, copyBack=True)
    np.save("pycuda_vne.npy", gvne.qube)
    np.save("pycuda_qseed.npy", gvne.qube)

    # the compute grid is allocated separately so we can reuse it
    cgrid = Grid.emptyAs(gridTemplate, nfields=4)

    # create an automaton and initialize the compute grid
    atm = AutomatonNNSimple()
    atm.Randomize(5.0, 4)

    atm.Initialize(cgrid, gqsd, gvne)
    # atm.Evolve(mol, cgrid, qref, maxiter=10, debug=True)

    # kernel = atm.WriteCode()
    # kernel = kernel.replace("PYCUDA_NX", str(cgrid.shape[1]))
    # kernel = kernel.replace("PYCUDA_NY", str(cgrid.shape[2]))
    # kernel = kernel.replace("PYCUDA_NZ", str(cgrid.shape[3]))
    # kernel = kernel.replace("PYCUDA_NPTS", str(cgrid.npts))
    # kernel = kernel.replace("PYCUDA_GRIDSTEP", str(cgrid.step))

    # ptx = SourceModule(kernel, include_dirs=[os.getcwd()])
    # cp, sb = ptx.get_global("cParams")
    # params = np.asarray(atm.params).astype(np.float32)
    # cuda.memcpy_htod(cp, params)  # copy constant param
    # ptx = ptx.get_function("gpu_automaton_nn_simple_evolve")
    # ptx.prepare([np.intp, np.intp])

    # ogrid = Grid.emptyAs(cgrid)
    # ptx.prepared_call(cgrid.GPUblocks, (8, 8, 8), cgrid.d_qube, ogrid.d_qube)

    # cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
    arr = np.zeros(gqsd.shape, order='F', dtype=np.float32)
    cuda.memcpy_dtoh(arr, gqsd.d_qube)
    arr = arr[0]
    # arr = np.ascontiguousarray(arr[0])
    print(arr.shape)

    # plt.imshow(cgrid.qube[0, :, :, 32].T, origin='lower')
    # plt.show()

    coords = mol.coords / ANG2BOR - np.array([gqsd.origin[c] for c in ['x', 'y', 'z']]) / ANG2BOR
    cell = (gqsd.step / ANG2BOR) * np.diag(gqsd.shape[1:])
    atoms = Atoms(numbers=mol.types, positions=coords, cell=cell, pbc=True)
    with open('test_gqsd.xsf', 'w') as f:
        write_xsf(f, [atoms], data=arr)
