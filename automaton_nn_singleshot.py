import numpy
import struct
import os

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from qmtools import Molecule, Grid, QMTools
from GAEngine import GAElement



class AutomatonSingleShot(GAElement):

	"""
		This is a neural network based automaton.
		The NN has 2+2 inputs: 2 for the current voxel and 2 for the neighoubring one.
		The inputs per voxel are the charge and the VNe.
		The output is the charge transfer between the voxels.
	"""

	# reads the kernel code as text
	fker = open('kernel_nn_ss.cu')
	sourceCode = fker.read()
	fker.close()


	def __init__(self):

		super().__init__()

		self.nlayers = 0

		# use float32 for this one
		self._precision = 4
		self._format = 'f'


	def Random(**kwargs):

		"""
			Create a randomised NN automaton with 16 neurons in each of the nl1 layers.
			An additional output layer will be added to take the 16 outputs of the nl1-th layer and
			calculate the final single output value (the charge transfer between neighbouring voxels).
			All random parameters are uniformly distributed between -pscale and +pscale.

			Expected kwargs:
				1. nlayers: number of NN layers [default=4]
				2. pscale: numerical range of the random parameters [default=1.0]
		"""


		atm = AutomatonSingleShot()

		atm.nlayers = kwargs.get('nlayers', 4)
		pscale = kwargs.get('pscale', 1.0)
		
		atm._dnasize = 16*16*atm.nlayers + 16*atm.nlayers + (16*1 + 1) # charge transfer NN
		atm._dnasize+= 12 # extra constants for the input layer to fill it up to 16 values

		atm.params = (2*numpy.random.random(atm._dnasize)-1)*pscale
		atm.params = atm.params.astype(numpy.float32)

		return atm



	def Mix(a,b):

		r = super(AutomatonSingleShot,AutomatonSingleShot).Mix(a,b)
		r.params = r.params.astype(numpy.float32)
		r.nlayers = a.nlayers

		return r

	# mutate is inherited


	def Evaluate(self, **kwargs):


		molecules = kwargs['molecules']

		fitness = 0
		
		debug = 	kwargs.get('debugElement', False)
		copyBack = 	kwargs.get('copyBack', False)
		
		# select some molecules at random from the batch
		nmols = kwargs.get('nEvalMols', 4)
		mols = numpy.random.choice(molecules, size=nmols, replace=False)

		for minfo in mols: # loop over all molecules in the batch

			cid = 	minfo['CID']
			mol = 	minfo['mol']
			vne = 	minfo['gvne']
			qref = 	minfo['qgrid']

			if debug:
				print("evaluating on CID {}".format(cid))

			# cgrid: compute grid for output
			# clear the compute grid - same shape as density grid but with 2 fields
			cgrid = Grid.emptyAs(qref, nfields=2)
			cuda.memcpy_htod(cgrid.d_qube, cgrid.qube)

			# copy vne(gpu) to cgrid(gpu)-offset npts (second field)
			ptr = int(cgrid.d_qube) + int(vne.qube.nbytes)
			cuda.memcpy_dtod(ptr, vne.d_qube, vne.qube.nbytes)



			# setup the kernel
			kernel = self.__WriteCode()

			# common substitutions
			kernel = kernel.replace('PYCUDA_NX', str(cgrid.shape[1]))
			kernel = kernel.replace('PYCUDA_NY', str(cgrid.shape[2]))
			kernel = kernel.replace('PYCUDA_NZ', str(cgrid.shape[3]))
			kernel = kernel.replace('PYCUDA_NPTS', str(cgrid.npts))
			kernel = kernel.replace('PYCUDA_GRIDSTEP', str(cgrid.step))
			
			if debug: # print the kernel for debug
				fout = open("atm.nnss.kernel.txt","w")
				fout.write(kernel)
				fout.close()

			# compile
			opts = []
			if debug: opts = ["--resource-usage"]
			ptx = SourceModule(kernel, include_dirs=[os.getcwd()], options=opts)

			# get the constant memory pointer
			cp, sb = ptx.get_global('cParams')
			cuda.memcpy_htod(cp, self.params) # copy constant params

			ptx = ptx.get_function("gpu_automaton_nn_singleshot")
			ptx.prepare([numpy.intp, numpy.intp])

			# output grid
			ogrid = Grid.emptyAs(cgrid, nfields=1)

			if debug: # debug save
				cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
				cgrid.SaveBINmulti('atm.{}.input'.format(cid),mol)
				qref.SaveBINmulti('atm.{}.qref'.format(cid),mol)
				numpy.save('atm.nnss.{}.input.npy'.format(cid), cgrid.qube)
				numpy.save('atm.nnss.{}.qref.npy'.format(cid), qref.qube)
				#print("qtot at start:", qtot)


			# compute q from VNe
			ptx.prepared_call(cgrid.GPUblocks, (8,8,8), cgrid.d_qube, ogrid.d_qube)

			# probably here we want to rescale it so the total is the total #ofElectrons of the molecule?
			qtot = QMTools.Compute_qtot(ogrid)
			factor = mol.qtot / qtot
			QMTools.Compute_qscale(ogrid, factor)

			# compare the ogrid to the refgrid (only the first field)
			qdiff = QMTools.Compute_qdiff(ogrid, qref, rel=False)

			if debug:
				print("automaton NN[SS] qdiff {}".format(qdiff))
				cuda.memcpy_dtoh(ogrid.qube, ogrid.d_qube)
				numpy.save('atm.nnss.{}.output.npy'.format(cid), ogrid.qube)
				ogrid.SaveBINmulti('atm.nnss.{}.output'.format(cid),mol)

			if copyBack:
				cuda.memcpy_dtoh(ogrid.qube, ogrid.d_qube)
				cgrid.qube = ogrid.qube
				cgrid.d_qube = ogrid.d_qube


			if numpy.isnan(qdiff):
				print("diff is nan!")
				qdiff = 9999

			fitness -= qdiff

		
		self.fitness = fitness / len(mols)
		return self.fitness






	## Writes the compute kernel for the genetic program.
	def __WriteCode(self):

		
		src = AutomatonSingleShot.sourceCode

		src = src.replace('PYCUDA_DNASIZE', str(self._dnasize))
		src = src.replace('PYCUDA_NL1', str(self.nlayers))
		src = src.replace('PYCUDA_NL2', str(0))

		src = src.replace('PYCUDA_ABS_A', str(0))
		src = src.replace('PYCUDA_ABS_B', str(0))

		return src




	### Compute fitness of this automaton.
	#
	## molecule: reference molecule
	## mol: molcule object
	## cgrid: computing grid, must be already initialised with q0 and vne in channels 0 and 1
	## qref: grid with the correct electron density of the molecule
	def SingleShot(self, mol, cgrid, qref, debug=False, copyBack=False):

		# setup the GP kernel
		kernel = self.WriteCode()

		# common substitutions
		kernel = kernel.replace('PYCUDA_NX', str(cgrid.shape[1]))
		kernel = kernel.replace('PYCUDA_NY', str(cgrid.shape[2]))
		kernel = kernel.replace('PYCUDA_NZ', str(cgrid.shape[3]))
		kernel = kernel.replace('PYCUDA_NPTS', str(cgrid.npts))
		
		if debug: # print the kernel for debug
			fout = open("atm.kernel.txt","w")
			fout.write(kernel)
			fout.close()

		# compile
		opts = []
		if debug: opts = ["--resource-usage"]
		ptx = SourceModule(kernel, include_dirs=[os.getcwd()], options=opts)

		# get the constant memory pointer
		cp, sb = ptx.get_global('cParams')
		params = numpy.asarray(self.params).astype(numpy.float32)
		cuda.memcpy_htod(cp, params) # copy constant params

		ptx = ptx.get_function("gpu_automaton_nn_singleshot")
		ptx.prepare([numpy.intp, numpy.intp])

		# output grid
		ogrid = Grid.emptyAs(cgrid)

		if debug: # debug save
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
			cgrid.SaveBINmulti('atm.input',mol)
			numpy.save('atm.nnss.input.npy', cgrid.qube)
			numpy.save('atm.nnss.qref.npy', qref.qube)
			#print("qtot at start:", qtot)


		# compute q from VNe
		ptx.prepared_call(cgrid.GPUblocks, (8,8,8), cgrid.d_qube, ogrid.d_qube)

		# compare the ogrid to the refgrid (only the first field)
		qdiff = QMTools.Compute_qdiff(ogrid, qref, rel=False)

		if debug:
			print("automaton NN[SS] qdiff {}".format(qdiff))
			cuda.memcpy_dtoh(ogrid.qube, ogrid.d_qube)
			numpy.save('atm.nnss.output.npy', ogrid.qube)

		if copyBack:
			cuda.memcpy_dtoh(ogrid.qube, ogrid.d_qube)
			cgrid.qube = ogrid.qube
			cgrid.d_qube = ogrid.d_qube


		if numpy.isnan(qdiff):
			print("diff is nan")
			return -999
			

		return -qdiff




	def CompareQ(self, cgrid, qref, rel=False):


		qdiff = QMTools.Compute_qdiff(cgrid, qref, rel)
		if numpy.isnan(qdiff):
			self.fitness = -9999 + numpy.random.random()
		else: 
			self.fitness = -qdiff

		return self.fitness

	
