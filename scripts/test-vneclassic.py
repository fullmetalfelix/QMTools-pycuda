
import numpy

from qmtools import BasisSet, Grid, Molecule, QMTools
from qmtools.cidhelper import CIDHelper

DATADIR = "data"


# load the basis set
basisset = BasisSet("../data/cc-pvdz.bin")


# get a random molecule
CID = numpy.random.choice(CIDHelper.CIDs)
print('downloading CID {}'.format(CID))

# fetch the info from the server
data = CIDHelper.GetMolecule(CID)

# create the molecule object
mol = Molecule(data['xyz'], data['D-CCSD'], basisset)
gridTemplate = Grid.DensityGrid(mol, 0.1, 3.0)
vgrid = QMTools.Compute_VNe_classic(gridTemplate, mol, divisions=16, copyBack=True)

numpy.save('vnetest.vne.{}.npy'.format(CID), vgrid.qube)

qgrid = QMTools.Compute_density(gridTemplate, mol, subgrid=2, copyBack=True)

numpy.save('vnetest.qref.{}.npy'.format(CID), qgrid.qube)

