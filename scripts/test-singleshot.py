import glob
import pickle
import sys

import numpy

from qmtools import BasisSet, Grid, Molecule
from qmtools.automaton_nn_singleshot import AutomatonSingleShot
from qmtools.GAEngine import GA

DATADIR = "data"


# load the basis set
basisset = BasisSet("../data/cc-pvdz.bin")



### create a batch of densities ###

def LoadBatch():

	files = glob.glob("{}/molecule.*.pickle".format(DATADIR))

	molecules = []

	for f in files:

		with open(f, 'rb') as handle:
			b = pickle.load(handle)

			mol = Molecule(b['xyz'], b['D-CCSD'], basisset)

			gridTemplate = Grid.DensityGrid(mol, 0.1, 3.0)
			print(b['CID'],gridTemplate.qube.shape)

			qgrid = Grid.emptyAs(gridTemplate)
			gvne = Grid.emptyAs(gridTemplate)

			qgrid.LoadFromNPY(b['qgrid'])
			gvne.LoadFromNPY(b['gvne'])

			b['mol'] = mol
			b['qgrid'] = qgrid
			b['gvne'] = gvne


			molecules.append(b)

	print("loaded {} molecules".format(len(molecules)))
	return molecules

### ########################### ###




### GA initialization ###

opts = GA.ReadParameters(sys.argv)
engine = GA(**opts)

engine.elementTemplate = AutomatonSingleShot
engine.elementInitDict = {'nlayers': 4}

engine.Initialize(noLog=True)

### ################# ###

### GA main cycle ### ###


molecules = LoadBatch()
evaldict = {'molecules': molecules, 'debug': True, 'debugElement': True}

fits = [e.fitness for e in engine.population]
fits = numpy.array(fits)

print(fits)
idxmax = numpy.argmax(fits)
print(idxmax)


engine.population[idxmax].Evaluate(**evaldict)


'''
for g in range(1000):

	engine.Evolve(**evaldict)

	usage += 1

	# every few generations create a new data batch for training
	if (usage % 30) == 0:
		mols = evaldict['molecules']
		newmols = MakeBatch(2, True)
		mols = mols[2:]
		mols.extend(newmols)
		evaldict['molecules'] = mols
		usage = 0
'''

### ################# ###

