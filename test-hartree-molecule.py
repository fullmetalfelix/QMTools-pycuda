from qmtools import BasisSet, Molecule, Grid, Automaton, QMTools
import numpy
import pycuda.driver as cuda
import struct

### test the hartree potential of a molecule

# load the basis set
basisset = BasisSet("cc-pvdz.bin")

# load a molecule
folder = "../qmtools/molecule_29766_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset)

# template grid - 1 field only
templateGrid = Grid.DensityGrid(mol, 0.02, 3.0)

# call density generator
qgrid = QMTools.Compute_density(templateGrid, mol, copyBack=True)
numpy.save('density_29766_0.02.npy', qgrid.qube)
#qgrid = Grid.emptyAs(templateGrid)
#qgrid.LoadData('density_29766_0.02.npy')


# make an equivalent grid for V
vgrid = Grid.emptyAs(qgrid)

# compute the hartree
QMTools.Compute_hartree(qgrid, mol, vgrid, copyBack=True)


linear = numpy.reshape(vgrid.qube, [qgrid.shape[1]*qgrid.shape[2]*qgrid.shape[3]])

fout = open("hartree_29766_0.02.bin","wb")

fout.write(struct.pack('i',mol.natoms))
for i in range(mol.natoms):
	fout.write(struct.pack('i',mol.types[i]))
	fout.write(struct.pack('f',mol.coords[i,0]))
	fout.write(struct.pack('f',mol.coords[i,1]))
	fout.write(struct.pack('f',mol.coords[i,2]))

fout.write(struct.pack('i',qgrid.shape[1]))
fout.write(struct.pack('i',qgrid.shape[2]))
fout.write(struct.pack('i',qgrid.shape[3]))

fout.write(struct.pack('f',vgrid.origin['x']))
fout.write(struct.pack('f',vgrid.origin['y']))
fout.write(struct.pack('f',vgrid.origin['z']))

for k in range(qgrid.shape[3]):
	for j in range(qgrid.shape[2]):
		for i in range(qgrid.shape[1]):
			fout.write(struct.pack('f',linear[i+j*qgrid.shape[1]+k*qgrid.shape[1]*qgrid.shape[2]]))

fout.close()
