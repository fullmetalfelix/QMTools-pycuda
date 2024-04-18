import numpy
import sys
import requests
import pickle
import glob


from qmtools.GAEngine import GA

from qmtools.automaton_nn_singleshot import AutomatonSingleShot

from qmtools.cidhelper import CIDHelper
from qmtools import BasisSet, Molecule, Grid, QMTools


DATADIR = "data"


# load the basis set
basisset = BasisSet("../data/cc-pvdz.bin")


	


### create a batch of densities ###

def MakeBatch(size=4, save=False):

	# download some random molecules
	# compute their density grid and save everything in a pickled dict in the DATADIR

	molecules = []

	for i in range(size):

		CID = numpy.random.choice(CIDHelper.CIDs)
		print('downloading CID {}'.format(CID))

		# fetch the info from the server
		data = CIDHelper.GetMolecule(CID)

		# create the molecule object
		mol = Molecule(data['xyz'], data['D-CCSD'], basisset)

		# compute the density from dmatrix
		print("computing density grid...")
		gridTemplate = Grid.DensityGrid(mol, 0.1, 3.0)
		print(gridTemplate.qube.shape)
		qgrid = QMTools.Compute_density(gridTemplate, mol, subgrid=2, copyBack=True)
		
		print("computing VNe")
		maxiters = 20*numpy.power(qgrid.npts,1.0/3)
		gvne = QMTools.Compute_VNe(gridTemplate, mol, adsorb=0.05, diff=0.02, tolerance=1.0e-9, maxiters=maxiters, copyBack=True)

		
		print("grids",gridTemplate.qube.shape, qgrid.qube.shape, gvne.qube.shape)

		data['mol'] = mol
		data['qgrid'] = qgrid
		data['gvne'] = gvne

		if save:

			dsave = dict(data)
			dsave['mol'] = None
			dsave['qgrid'] = qgrid.qube
			dsave['gvne'] = gvne.qube

			with open('{}/molecule.{}.pickle'.format(DATADIR, data['CID']), 'wb') as handle:
				pickle.dump(dsave, handle, protocol=pickle.HIGHEST_PROTOCOL)


		molecules.append(data)

	return molecules

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

engine.Initialize()

### ################# ###

### GA main cycle ### ###


evaldict = {'molecules': [], 'debug': True}

# check if we got molecules in the cache
molecules = LoadBatch()
evaldict['molecules'] = molecules

usage = 0
if len(molecules) < 10:
	print("downloading new molecules...")
	newmols = MakeBatch(10, True)
	molecules.extend(newmols)
	evaldict['molecules'] = molecules



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


### ################# ###

