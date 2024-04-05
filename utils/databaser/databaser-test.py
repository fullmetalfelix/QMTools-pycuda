from databaser import BasisSet, Molecule, DensityCalculator
import numpy
import glob
import math
import sys


basisset = BasisSet("../../cc-pvdz.bin")


moldirs = glob.glob('../molecule_*')
moldirs.sort()

#print(len(moldirs))
#print(len(moldirs) / 24)
#print(sys.argv)

batch = int(sys.argv[1])
total = int(sys.argv[2])

ndata = math.ceil(len(moldirs) / total)
d0 = ndata*batch
df = min(ndata*(batch+1), len(moldirs))

moldirs = moldirs[d0 : df]
#print(len(moldirs))



for folder in moldirs:
	print(folder)
	
	CID = int(folder.split('_')[1])
	fdm = folder + "/D-CCSD.npy"
	fxyz = folder + "/GEOM-B3LYP.xyz"
	
	mol = Molecule(fxyz, fdm, basisset)
	mol.CID = CID

	data = DensityCalculator.Compute(mol, basisset)
	numpy.save('molecule_{}.npy'.format(CID), data)


