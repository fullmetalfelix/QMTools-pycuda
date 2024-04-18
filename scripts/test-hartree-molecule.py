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

stp = 0.1

vgrid = Grid.DensityGrid(mol, stp, fat=3.0)
QMTools.Compute_hartree(vgrid, mol, copyBack=True)

print("saving bin...")
vgrid.SaveBIN('hartree-GTO_{}.bin'.format(stp), mol)
print(vgrid.qube[0,0,0,0], numpy.mean(vgrid.qube))



'''
sg = 2
for i in range(10,3,-5):

	stp = 0.01*i
	print("step is",stp)

	# template grid - 1 field only
	#templateGrid = Grid.DensityGrid(mol, stp, 10.0)

	#for sg in range(2,3):

	# call density generator
	#qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=sg, copyBack=True)
	#qgrid.SaveBIN('density-large_29766_{}_{}.bin'.format(stp,sg), mol)
	#fname = 'density_29766_{}_2.bin'.format(stp)
	qgrid = Grid.LoadBIN('density-large_29766_{}_{}.bin'.format(stp,sg))

	#print(qgrid)

	

	# compute the hartree
	vgrid = QMTools.Compute_hartree(qgrid, mol, tolerance=0.001, copyBack=True)
	vgrid.SaveBIN('hartree-large_dgrid_{}_SG2.bin'.format(stp), mol)
'''

'''
qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=2, copyBack=True)
qgrid.SaveBIN('density_29766_0.1_2.bin', mol)
numpy.save(   'density_29766_0.1_2.npy', qgrid.qube)

qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=4, copyBack=True)
qgrid.SaveBIN('density_29766_0.1_4.bin', mol)
numpy.save(   'density_29766_0.1_4.npy', qgrid.qube)


templateGrid = Grid.DensityGrid(mol, 0.05, 3.0)

qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=1, copyBack=True)
qgrid.SaveBIN('density_29766_0.05_1.bin', mol)
numpy.save(   'density_29766_0.05_1.npy', qgrid.qube)

qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=2, copyBack=True)
qgrid.SaveBIN('density_29766_0.05_2.bin', mol)
numpy.save(   'density_29766_0.05_2.npy', qgrid.qube)


templateGrid = Grid.DensityGrid(mol, 0.025, 3.0)

qgrid = QMTools.Compute_density(templateGrid, mol, subgrid=1, copyBack=True)
qgrid.SaveBIN('density_29766_0.025_1.bin', mol)
numpy.save(   'density_29766_0.025_1.npy', qgrid.qube)
'''


#qgrid = Grid.emptyAs(templateGrid)
#qgrid.LoadData('density_29766_0.02.npy')




'''
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


linear = numpy.reshape(qgrid.qube, [qgrid.shape[1]*qgrid.shape[2]*qgrid.shape[3]])

fout = open("density_29766_0.02.bin","wb")

fout.write(struct.pack('i',mol.natoms))
for i in range(mol.natoms):
	fout.write(struct.pack('i',mol.types[i]))
	fout.write(struct.pack('f',mol.coords[i,0]))
	fout.write(struct.pack('f',mol.coords[i,1]))
	fout.write(struct.pack('f',mol.coords[i,2]))

fout.write(struct.pack('i',qgrid.shape[1]))
fout.write(struct.pack('i',qgrid.shape[2]))
fout.write(struct.pack('i',qgrid.shape[3]))

fout.write(struct.pack('f',qgrid.origin['x']))
fout.write(struct.pack('f',qgrid.origin['y']))
fout.write(struct.pack('f',qgrid.origin['z']))

for k in range(qgrid.shape[3]):
	for j in range(qgrid.shape[2]):
		for i in range(qgrid.shape[1]):
			fout.write(struct.pack('f',linear[i+j*qgrid.shape[1]+k*qgrid.shape[1]*qgrid.shape[2]]))

fout.close()
'''