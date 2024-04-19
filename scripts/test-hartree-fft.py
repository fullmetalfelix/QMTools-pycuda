
import time

import matplotlib.pyplot as plt
import numpy

from qmtools import BasisSet, Grid, Molecule, QMTools

# load the basis set
basisset = BasisSet("../data/cc-pvdz.bin")

# load a molecule
folder = "../data/molecule_29766_0/"
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

plt.figure(figsize=(8, 8))
for sigma in [0.5, 0.3, 0.2, 0.1, 0.05, 0.03]:

	t0 = time.perf_counter()
	vqube = QMTools.Compute_hartree_fft(qgrid, mol, sigma=sigma, cutoff=6)
	t1 = time.perf_counter()
	print('Hartree FFT time:', t1 - t0)

	plt.plot(vqube[:, 116, 64], label=f'sigma = {sigma}Å')

plt.legend()
plt.savefig('pot_test.png', dpi=200)
plt.close()

numpy.save(f'potential_cuda_{step}_{sg}.npy', vqube)
