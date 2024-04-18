import numpy
import pycuda.driver as cuda

from qmtools import Grid, QMTools

### test the hartree potential of a ~point charge
# TODO: convert this to work with a molecule instead

#basisset = BasisSet("../data/cc-pvdz.bin")
qm = QMTools()


# grid for fake charge density
qgrid = Grid([8*8,8*8,8*8], 0.1)

# create a fake charge density - one e in the middle
linear = numpy.reshape(qgrid.qube, [qgrid.shape[1]*qgrid.shape[2]*qgrid.shape[3]])
i = j = k = 4*8
linear[i + j*qgrid.shape[1] + k*qgrid.shape[1]*qgrid.shape[2]] = 1.0
linear = linear.astype(numpy.float32)
cuda.memcpy_htod(qgrid.d_qube, linear)

#qgrid.qube[0,0,0,4*8] = 1.0
#cuda.memcpy_htod(qgrid.d_qube, qgrid.qube) # copy to GPU


# make an equivalent grid for V
#vgrid = Grid.emptyAs(qgrid)


# compute the hartree
vgrid = QMTools.Compute_hartree(qgrid, None, tolerance=1.0e-16, copyBack=True)
vgrid.SaveBIN('hartree-point.bin')

linear = numpy.reshape(vgrid.qube, [qgrid.shape[1]*qgrid.shape[2]*qgrid.shape[3]])

fout = open("hartree-point.out","w")
for i in range(qgrid.shape[1]):

	idx = i + 4*8*qgrid.shape[1] + 4*8*qgrid.shape[1]*qgrid.shape[2]
	fout.write("{} {}\n".format(qgrid.step*i+qgrid.step/2, linear[idx]))

fout.close()
