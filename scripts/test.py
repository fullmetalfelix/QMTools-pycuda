import numpy

from qmtools import Automaton, BasisSet, Grid, Molecule, QMTools

basisset = BasisSet("../data/cc-pvdz.bin")
qm = QMTools()

folder = "../data/molecule_29766_0/"
mol = Molecule(folder+"GEOM-B3LYP.xyz", folder+"D-CCSD.npy", basisset)

gridTemplate = Grid.DensityGrid(mol, 0.1, 3.0)

# test the density generator
#qref = QMTools.Compute_density(gridTemplate, mol, copyBack=True)
#numpy.save('pycuda_qref.npy', qref.qube)


#'''
qref = Grid.emptyAs(gridTemplate)
qref.LoadData('pycuda_qref.npy')

gvne = QMTools.Compute_VNe(gridTemplate, mol, adsorb=0.1, diff=0.01, tolerance=0.000000001, copyBack=True)
gqsd = QMTools.Compute_qseed(gridTemplate, mol, copyBack=True)
numpy.save('pycuda_vne.npy', gvne.qube)
numpy.save('pycuda_qseed.npy', gvne.qube)
#'''

# the compute grid is allocated separately so we can reuse it
cgrid = Grid.emptyAs(gridTemplate, nfields=4)

# create an automaton and initialize the compute grid
atm = Automaton()
atm.Randomize(10.0, 2)
#print(numpy.max(cgrid.qube[0]), numpy.max(cgrid.qube[1]), numpy.max(cgrid.qube[2]), numpy.max(cgrid.qube[3]))
#print(numpy.sum(cgrid.qube[0]), numpy.sum(cgrid.qube[1]), numpy.sum(cgrid.qube[2]), numpy.sum(cgrid.qube[3]))

binary = atm.Binarize()
#print(binary)

atm.Randomize(0.0,2)
atm.LoadBinary(binary)

binary2 = atm.Binarize()

print(len(binary),len(binary2))

for c in range(len(binary)):
	if binary[c] != binary2[c]:
		print("mismatch",c,binary[c],binary2[c])



atm.Mutate(0.1,0.1,16.0)


atm.Initialize(cgrid, gqsd, gvne)
atm.Evolve(mol, cgrid)


#print(atm.Binarize())


#mgrid = Grid.emptyAs(qref, nfields=4)
#mgrid.Compute_qseed(mol, copyBack=True)
#numpy.save("pycuda_qseed.npy",mgrid.qube[0])
#mgrid.Compute_VNe(mol)

#qref.ComputeDensity_subgrid(mol)
#numpy.save("density_29766_pycuda_sh.npy",qref.qube)

#mol.ComputeVNe(qref)

