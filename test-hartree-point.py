from qmtools import BasisSet, Molecule, Grid, Automaton, QMTools
import numpy
import pycuda.driver as cuda
from pycuda import gpuarray



### test the hartree potential of a ~point charge


#basisset = BasisSet("cc-pvdz.bin")
qm = QMTools()


# grid for fake charge density
qgrid = Grid([4*8,8,8],0.01)

# create a fake charge density - one e in the middle
qgrid.qube[0,0,0,0] = 1.0
cuda.memcpy_htod(qgrid.d_qube, qgrid.qube) # copy to GPU


# make an equivalent grid for V
vgrid = Grid.emptyAs(qgrid)


# compute the hartree
QMTools.Compute_hartree_fromGrid(qgrid, vgrid, copyBack=True)


linear = numpy.reshape(vgrid.qube, [qgrid.shape[1]*qgrid.shape[2]*qgrid.shape[3]])

fout = open("hartree-point.out","w")
for i in range(qgrid.shape[1]):

	fout.write("{} {}\n".format(qgrid.step*i+qgrid.step/2, linear[i]))

fout.close()
