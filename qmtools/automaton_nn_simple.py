import struct

import numpy
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from qmtools import CUDA_SRC_DIR, Grid, QMTools

# this must implement a few functions to work with the GAEngine
# Random = return a new random automaton
# Mix = return a new automaton by mixing two parents
# Mutate = mutate the element
# Evaluate = perform fitness evaluation



class Automaton:

	"""
	This is a neural network based automaton.
	The NN has 2+2 inputs: 2 for the current voxel and 2 for the neighoubring one.
	The inputs per voxel are the charge and the VNe.
	The output is the charge transfer between the voxels.

	There is no extra information stored in the grid (no A/B fields)
	

	"""


	def __init__(self):

		self.params = []
		self.nlayers = 0
		self.fitness = 0
		self.dnasize = 0
		self.fitness = -9999



	def Random(**kwargs):

		"""
		
		Create a randomised NN automaton with 16 neurons in each of the nl1 layers.
		An additional output layer will be added to take the 16 outputs of the nl1-th layer and
		calculate the final single output value (the charge transfer between neighbouring voxels).
		All random parameters are uniformly distributed between -pscale and +pscale.

		"""


		atm = Automaton()

		atm.nlayers = kwargs.get('nlayers', 4)
		pscale = kwargs.get('pscale', 1.0)
		
		atm.dnasize = 16*16*atm.nlayers + 16*atm.nlayers + (16*1 + 1) # charge transfer NN
		atm.dnasize+= 12 # extra constants for the input layer to fill it up to 16 values

		atm.params = (2*numpy.random.random(atm.dnasize)-1)*pscale
		atm.params = atm.params.astype(numpy.float32)

		return atm





	def Initialize(self, cgrid, q0, vne):

		# clear the compute grid
		cgrid.qube *= 0
		cuda.memcpy_htod(cgrid.d_qube, cgrid.qube)

		# copy q0(gpu) to cgrid(gpu)-beginning -- make sure to copy only one field
		cuda.memcpy_dtod(cgrid.d_qube, q0.d_qube, int(q0.npts * 4))

		# copy vne(gpu) to cgrid(gpu)-offset npts
		ptr = int(cgrid.d_qube) + int(q0.npts * 4) #int(vne.qube.nbytes)
		cuda.memcpy_dtod(ptr, vne.d_qube, vne.qube.nbytes)




	## Writes the compute kernel for the genetic program.
	def WriteCode(self):

		
		src = QMTools.srcAutomatonNN

		src = src.replace('PYCUDA_DNASIZE', str(self.dnasize))
		src = src.replace('PYCUDA_NL1', str(self.nlayers[0]))
		src = src.replace('PYCUDA_NL2', str(0))

		src = src.replace('PYCUDA_ABS_A', str(0))
		src = src.replace('PYCUDA_ABS_B', str(0))

		return src


	## Save the automaton parameters in a bytes string.
	def Binarize(self):

		bs = b''

		for i in self.params:
			bs += struct.pack('f', i)
		
		bs += struct.pack('f', self.fitness)

		return bs


	## Load the automaton code from a binary string.
	# The automaton code topology has to be already set
	# returns the offset of the next byte to read from barray
	def LoadBinary(self, barray):

		ptr = 0
		
		# read the big params
		for i in range(self.params.shape[0]):
			self.params[i] = struct.unpack('f', barray[ptr:ptr+4])[0]
			ptr += 4

		self.fitness = struct.unpack('f', barray[ptr:ptr+4])[0]
		ptr += 4

		print('automaton NN simple loaded [fitness={}]'.format(self.fitness))
		return ptr




	### Compute fitness of this automaton.
	#
	## molecule: reference molecule
	## mol: molcule object
	## cgrid: computing grid, must be already initialised with q0 and vne in channels 0 and 1
	## qref: grid with the correct electron density of the molecule
	def Evolve(self, mol, cgrid, qref, maxiter=1000, tolerance=0.1, debug=False, copyBack=False):

		# setup the GP kernel
		kernel = self.WriteCode()

		# common substitutions
		kernel = kernel.replace('PYCUDA_NX', str(cgrid.shape[1]))
		kernel = kernel.replace('PYCUDA_NY', str(cgrid.shape[2]))
		kernel = kernel.replace('PYCUDA_NZ', str(cgrid.shape[3]))
		kernel = kernel.replace('PYCUDA_NPTS', str(cgrid.npts))
		
		if debug: # print the kernel for debug
			fout = open("atm.gans.kernel.txt","w")
			fout.write(kernel)
			fout.close()

		# compile
		opts = []
		if debug: opts = ["--resource-usage"]
		ptx = SourceModule(kernel, include_dirs=[str(CUDA_SRC_DIR)], options=opts)

		# get the constant memory pointer
		cp, sb = ptx.get_global('cParams')
		params = numpy.asarray(self.params).astype(numpy.float32)
		cuda.memcpy_htod(cp, params) # copy constant params


		ptx = ptx.get_function("gpu_automaton_nn_simple_evolve")
		ptx.prepare([numpy.intp, numpy.intp])

		ogrid = Grid.emptyAs(cgrid)
		
		if debug: # debug save
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
			#cgrid.SaveBINmulti('atm.gans.input',mol)
			numpy.save('atm.gans.input.npy', cgrid.qube)
			qtot = QMTools.Compute_qtot(cgrid)
			print("qtot at start:", qtot)


		# do evolution
		qdiff = tolerance + 1
		rep = 0
		lastqdiff = 0
		convstrikes = 0
		penalty = 0

		while (qdiff > tolerance or rep < 10) and rep < maxiter:

			# propagate q,A,B
			ptx.prepared_call(cgrid.GPUblocks, (8,8,8), cgrid.d_qube, ogrid.d_qube)

			# switch pointers
			tmp = cgrid.d_qube
			cgrid.d_qube = ogrid.d_qube
			ogrid.d_qube = tmp

			# renormalize q if needed
			if rep % 100 == 0:
				qtot = QMTools.Compute_qtot(cgrid)
				if numpy.abs(qtot-mol.qtot) > 1.0e-3:
					factor = float(mol.qtot) / qtot
					QMTools.Compute_qscale(cgrid, factor)

				#cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
				#numpy.save('atm.output-{}.npy'.format(rep+1), cgrid.qube)

			if debug and rep % 100 == 0:
				qtot = QMTools.Compute_qtot(cgrid)
				cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
				#cgrid.SaveBINmulti('atm.output-{}'.format(rep+1),mol)
				numpy.save('atm.gans.output-{}.npy'.format(rep+1), cgrid.qube)

			# check if q converged - compute difference between current q grid and previous
			# check if it is converging towards the right solution!
			qdiff = QMTools.Compute_qdiff(cgrid, qref, rel=False)
			penalty += qdiff
			if debug: print("iteration {}: penalty {} qdiff {} qtot {}".format(rep, penalty/(rep+1), qdiff, qtot))
			#print("iteration {}: penalty {} qdiff {}".format(rep, penalty, qdiff))

			if numpy.isnan(qdiff):
				print("diff is nan")
				penalty *= 2
				break
			
			# we have to punish non-convergent behaviour
			# qdiff should become smaller over time
			if qdiff >= lastqdiff:
				convstrikes += 1
			else: # if qdiff decreased, reduce the penalty
				convstrikes = 0

			rep += 1

			# with enough strikes the automaton is OUT!
			if convstrikes == 15:
				penalty *= 1.2
				break

			lastqdiff = qdiff
			

		if copyBack:
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)

		if debug:
			qtot = QMTools.Compute_qtot(cgrid)
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
			numpy.save('atm.gans.output-{}.npy'.format(rep+1), cgrid.qube)

		penalty /= rep
		return True, penalty




	def Evaluate(self, mol, cgrid, qref, steps=10, debug=False, copyBack=False):

		# setup the GP kernel
		kernel = self.WriteCode()

		# common substitutions
		kernel = kernel.replace('PYCUDA_NX', str(cgrid.shape[1]))
		kernel = kernel.replace('PYCUDA_NY', str(cgrid.shape[2]))
		kernel = kernel.replace('PYCUDA_NZ', str(cgrid.shape[3]))
		kernel = kernel.replace('PYCUDA_NPTS', str(cgrid.npts))
		
		if debug: # print the kernel for debug
			fout = open("atm.gans.kernel.txt","w")
			fout.write(kernel)
			fout.close()

		# compile
		opts = []
		if debug: opts = ["--resource-usage"]
		ptx = SourceModule(kernel, include_dirs=[str(CUDA_SRC_DIR)], options=opts)

		# get the constant memory pointer
		cp, sb = ptx.get_global('cParams')
		params = numpy.asarray(self.params).astype(numpy.float32)
		cuda.memcpy_htod(cp, params) # copy constant params


		ptx = ptx.get_function("gpu_automaton_nn_simple_evolve")
		ptx.prepare([numpy.intp, numpy.intp])

		ogrid = Grid.emptyAs(cgrid)
		
		if debug: # debug save
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
			#cgrid.SaveBINmulti('atm.gans.input',mol)
			numpy.save('atm.gans.input.npy', cgrid.qube)
			qtot = QMTools.Compute_qtot(cgrid)
			print("qtot at start:", qtot)


		rep = 0
		lastqdiff = 0
		convstrikes = 0
		penalty = 0

		for rep in range(steps):

			# setup the cgrid with some initial q
			# TODO: ...

			# make sure the VNe is always there?


			# compute how bad the initial guess is
			qdiff0 = QMTools.Compute_qdiff(cgrid, qref, rel=False)

			# propagate q
			ptx.prepared_call(cgrid.GPUblocks, (8,8,8), cgrid.d_qube, ogrid.d_qube)
			# now ogrid has the updated charge


			if debug:
				cuda.memcpy_dtoh(ogrid.qube, ogrid.d_qube)
				numpy.save('atm.gans.output-{}.npy'.format(rep+1), ogrid.qube)

			# check if q converged - compute difference between current q grid and previous
			# check if it is converging towards the right solution!
			qdiff = QMTools.Compute_qdiff(ogrid, qref, rel=False)
			deltaqdiff = qdiff - qdiff0

			if debug: print("iteration {}: penalty {} improvement {}".format(rep, penalty, deltaqdiff))

			# deltaqdiff < 0 => the solution improved the initial guess
			# deltaqdiff >=0 => no improvement
			# ideally we want qdiff < qdiff0 i.e. the automaton made the solution better

			penalty += numpy.exp(deltaqdiff)

			

		if copyBack:
			cuda.memcpy_dtoh(cgrid.qube, ogrid.d_qube)


		penalty /= steps
		return True, penalty


	def CompareQ(self, cgrid, qref, rel=False):


		qdiff = QMTools.Compute_qdiff(cgrid, qref, rel)
		if numpy.isnan(qdiff):
			self.fitness = -9999 + numpy.random.random()
		else: 
			self.fitness = -qdiff

		return self.fitness


	def Mix(a,b):

		r = a.__class__()
		r.nlayers = a.nlayers
		r.params = numpy.copy(a.params)
		r.dnasize = a.dnasize

		for i in range(a.params.shape[0]):

			x = numpy.random.random()
			if x < 0.5: r.params[i] = a.params[i]
			else: r.params[i] = b.params[i]

		return r


	def Mutate(self, amount, mscale, pscale):

		mutagen = (2*numpy.random.random(self.dnasize)-1)*mscale
		mutagen = mutagen.astype(numpy.float32)

		mask = numpy.random.random(self.dnasize).astype(numpy.float32)
		mask[mask>amount] = 0

		self.params += mutagen*mask
