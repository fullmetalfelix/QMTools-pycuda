
from qmtools import BasisSet, Molecule, Grid, QMTools
import numpy
import time

# load the basis set
basisset = BasisSet("cc-pvdz.bin")

# load a molecule
folder = "./molecule_29766_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset)

step = 0.05
sg = 2

t0 = time.perf_counter()
templateGrid = Grid.DensityGrid(mol, step, 3.0)
t1 = time.perf_counter()
print('Grid allocation time:', t1 - t0)

# t0 = time.perf_counter()
# qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=sg, copyBack=True)
# t1 = time.perf_counter()
# qgrid.SaveBIN('density_29766_{}_{}.bin'.format(step,sg), mol)
# print('Density computation time:', t1 - t0)

qgrid = Grid.LoadBIN('density_29766_0.05_2.bin')

print(qgrid.qube.sum())

t0 = time.perf_counter()
vqube = QMTools.Compute_hartree_fft(qgrid, mol, sigma=0.5, cutoff=6)
t1 = time.perf_counter()
print('Hartree FFT time:', t1 - t0)

numpy.save(f'potential_cuda_{step}_{sg}.npy', vqube)
