import numpy
from ctypes import *
import pickle
import struct
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import os
from enum import IntEnum, unique

import skcuda.fft as cufft

ANG2BOR = 1.8897259886

MAXAOS = 15
MAXAOC = 20
MOLFAT = 2.0

sizeofInt = 4
sizeofFloat = 4
sizeofDouble = 8


## Stores the basis set information, also on the GPU.
class BasisSet:


	def __init__(self, filename):

		print("reading basis set from: ", filename)
		ncmax = 0
		maxorbs = 0
		
		fbin = open(filename, 'rb')
		
		# number of atoms in this basis set
		natm = struct.unpack("i",fbin.read(sizeofInt))[0]

		self.atomOffset = numpy.zeros(100).astype(numpy.int32); self.atomOffset -= 1
		self.nshells = numpy.zeros(MAXAOS*100).astype(numpy.int32)
		
		self.shellOffset 	= numpy.zeros(MAXAOS*100).astype(numpy.int32)
		self.Ls 			= numpy.zeros(MAXAOS*100).astype(numpy.int32)
		self.alphas 		= numpy.zeros(MAXAOS*MAXAOC*100).astype(numpy.float32)
		self.coeffs 		= numpy.zeros(MAXAOS*MAXAOC*100).astype(numpy.float32)

		self.global_m_values = numpy.asarray([[0,0,0,0,0],[0,1,-1,0,0],[0,1,-1,2,-2]]).astype(numpy.int32)
		
		offset = 0		
		orbcount = 0

		for i in range(natm):

			# read the Z of this atom
			z = struct.unpack("i",fbin.read(sizeofInt))[0]
			self.atomOffset[z] = orbcount

			# read the shells info
			tmpi = struct.unpack("i",fbin.read(sizeofInt))[0]
			self.nshells[z] = tmpi
			maxorbs = numpy.max([tmpi, maxorbs])

			# maybe we want to keep parameters only for the ones we actually use?
			for j in range(self.nshells[z]):

				# read the L of this shell
				self.Ls[orbcount] = struct.unpack('i',fbin.read(sizeofInt))[0]
				
				# number of primitives in the contraction
				nc = struct.unpack('i',fbin.read(sizeofInt))[0]
				
				self.shellOffset[orbcount] = offset

				buffer = struct.unpack("d"*nc, fbin.read(sizeofDouble*nc))
				buffer = numpy.asarray(buffer).astype(numpy.float32)
				self.alphas[offset:offset+nc] = buffer

				buffer = struct.unpack("d"*nc, fbin.read(sizeofDouble*nc))
				buffer = numpy.asarray(buffer).astype(numpy.float32)
				self.coeffs[offset:offset+nc] = buffer

				#printf("B%i\tAO%i\n", i, j);
				#printf("Z=%03i AO%02i\tl=%i nc=%i\n",z, j, basisset->Ls[orbcount], nc);
				
				ncmax = numpy.max([nc,ncmax])

				orbcount += 1
				offset += MAXAOC
			# end of j loop
		#end of i loop

		self.nparams = offset

		fbin.close()

		# allocate in cuda
		self.d_alphas = cuda.mem_alloc(self.alphas.nbytes)
		self.d_coeffs = cuda.mem_alloc(self.coeffs.nbytes)

		cuda.memcpy_htod(self.d_alphas, self.alphas)
		cuda.memcpy_htod(self.d_coeffs, self.coeffs)

		print("basis set read - maxorbs={}  ncmax={} - nparams={}\n".format(maxorbs, ncmax, offset))

## Stores a molecule in BOHR, also on the GPU.
class Molecule:


	def __init__(self, filexyz, filedm, basisset):

		print("opening molecule (ang):", filexyz)
		self.basisset = basisset

		m = numpy.loadtxt(filexyz)
		
		# make coordinates in bohr
		m[:,1:] *= ANG2BOR
		self.types  = numpy.asarray(m[:,0], dtype=numpy.int32)
		self.coords = numpy.asarray(m[:,1:], dtype=numpy.float32)
		self.natoms = numpy.int32(self.types.shape[0])
		# total nuclear charge
		self.qtot = numpy.int32((numpy.sum(self.types)))


		# load the DM
		print("opening density matrix:", filedm)
		dm = numpy.load(filedm)
		self.norbs = numpy.int32(dm.shape[0])
		self.dm = numpy.asarray(dm, dtype=numpy.float32)
		print("# of orbitals:",self.norbs)
		# make the basis
		self.ALMOs = numpy.zeros((self.norbs, 4)).astype(numpy.int16)

		c = 0
		for a in range(self.natoms):

			Z = self.types[a]
			atomOS = basisset.atomOffset[Z]
			nsh = basisset.nshells[Z]
			#print(Z, atomOS, nsh)

			for s in range(nsh):

				L = basisset.Ls[atomOS+s]
				
				# loop over m values
				for mi in range(2*L+1):
					
					self.ALMOs[c,0] = a
					self.ALMOs[c,1] = L
					self.ALMOs[c,2] = basisset.global_m_values[L,mi]
					self.ALMOs[c,3] = basisset.shellOffset[atomOS+s]
					c += 1


		self.d_types = cuda.mem_alloc(self.types.nbytes); cuda.memcpy_htod(self.d_types, self.types)
		self.d_coords = cuda.mem_alloc(self.coords.nbytes); cuda.memcpy_htod(self.d_coords, self.coords)

		self.d_dm = cuda.mem_alloc(self.dm.nbytes); cuda.memcpy_htod(self.d_dm, self.dm)
		self.d_ALMOs = cuda.mem_alloc(self.ALMOs.nbytes); cuda.memcpy_htod(self.d_ALMOs, self.ALMOs)



	def Translate(self, dx=0,dy=0,dz=0):

		self.coords[:,0] += dx
		self.coords[:,1] += dy
		self.coords[:,2] += dz

		cuda.memcpy_htod(self.d_coords, self.coords)


	# end of Molecule class

# Multifield grid.
class Grid:


	def __init__(self, shape, step, origin=[0,0,0]):

		if len(shape) == 3:
			print('WARNING: augmenting the grid shape!')
			shape = (1,shape[0],shape[1],shape[2])

		if len(shape) != 4:
			raise ValueError("shape has to be 4-dimensional: nfields, nx, ny, nz")
		

		self.shape = tuple(shape)

		grd = numpy.asarray(shape).astype(numpy.int32)[1:]
		blk = numpy.floor(grd/8).astype(numpy.int32)
		if not numpy.array_equal(grd,blk*8):
			raise ValueError("shape for xyz have to be multiple of 8")
		self.GPUblocks = tuple([x.item() for x in blk])

		self.qube = numpy.zeros(self.shape).astype(numpy.float32)
		self.d_qube = cuda.mem_alloc(self.qube.nbytes); cuda.memcpy_htod(self.d_qube, self.qube)

		self.npts = numpy.uint32(self.shape[1]*self.shape[2]*self.shape[3])
		self.nfields = numpy.uint32(self.shape[0])

		self.step = numpy.float32(step*ANG2BOR)
		self.origin = gpuarray.vec.make_float3(origin[0],origin[1],origin[2])
		self.lattice = numpy.asarray([[1,0,0],[0,1,0],[0,0,1]], dtype=numpy.float32)

	def __str__(self):

		return "origin: {}\nshape: {} -- points: {} gpublocks: {}".format(
			self.origin, self.shape, self.npts, self.GPUblocks
		)		

	### Get a tuple with the spatial shape of the grid (without the n. of fields)
	def space_shape(self):

		return (self.shape[1], self.shape[2], self.shape[3])

	def space_shape_uint3(self):

		return gpuarray.vec.make_uint3(self.shape[1], self.shape[2], self.shape[3])



	### Creates a cartesian grid around a molecule.
	def DensityGrid(molecule, step, fat):

		# get the min and max of each coordinate with some fat
		xyz = numpy.array(molecule.coords, copy=True)
		crdmax = numpy.amax(xyz, axis=0) + fat*ANG2BOR
		crdmin = numpy.amin(xyz, axis=0) - fat*ANG2BOR

		#print("density grid min {} -- max {}".format(crdmin, crdmax))

		grd = crdmax - crdmin
		grd = grd / (step * ANG2BOR)
		grd = grd / 8.0;
		grd = numpy.ceil(grd).astype(numpy.uint32) * 8
		
		shape = (1, grd[0], grd[1], grd[2])

		grid = Grid(shape, step)
		grid.origin = gpuarray.vec.make_float3(crdmin[0],crdmin[1],crdmin[2])
		return grid



	### Create a grid with the same shape and step as the given one.
	### The number of fields can be optionally changed.
	### New grid will be zero everywhere.
	def emptyAs(g, nfields=None):

		shape = g.shape
		if nfields != None:
			shape = (nfields, shape[1], shape[2], shape[3])

		grid = Grid(shape, 1.0)
		grid.origin = g.origin
		grid.lattice = g.lattice
		grid.step = g.step
		return grid


	### Loads the qube from an npy file.
	### The grid has to be set with the same shape as the saved npy.
	def LoadData(self, filename):

		self.qube = numpy.load(filename)
		cuda.memcpy_htod(self.d_qube, self.qube)


	### Save the whole grid object.
	def Save(self, filename):

		d = {}
		d["shape"] = self.shape
		d["origin"] = self.origin
		d["step"] = self.step
		d["lattice"] = self.lattice
		d["qube"] = self.qube
		
		pickle.dump(d, open(filename, "wb"))
		
	### Restore a grid from saved file.
	def Load(molecule, filename):

		d = pickle.load(open(filename, "rb"))

		grid = Grid(d["shape"], 0.1)

		grid.origin = d["origin"]
		grid.step = d["step"]
		grid.lattice = d["lattice"]
		grid.qube = d["qube"]
		cuda.memcpy_htod(grid.d_qube, grid.qube)

		return grid


	### Save in binary format, optionally with a molecule info
	### This only saves the first field
	def SaveBIN(self, filename, mol=None):


		linear = numpy.reshape(self.qube, [self.shape[1]*self.shape[2]*self.shape[3]])

		fout = open(filename,"wb")

		if mol == None:
			fout.write(struct.pack('i',0))
		else:
			fout.write(struct.pack('i',mol.natoms))
			for i in range(mol.natoms):
				fout.write(struct.pack('i',mol.types[i]))
				fout.write(struct.pack('f',mol.coords[i,0]))
				fout.write(struct.pack('f',mol.coords[i,1]))
				fout.write(struct.pack('f',mol.coords[i,2]))

		fout.write(struct.pack('i',self.shape[1]))
		fout.write(struct.pack('i',self.shape[2]))
		fout.write(struct.pack('i',self.shape[3]))

		fout.write(struct.pack('f',self.step))
		fout.write(struct.pack('f',self.origin['x']))
		fout.write(struct.pack('f',self.origin['y']))
		fout.write(struct.pack('f',self.origin['z']))

		fout.write(linear.tobytes())

		#for k in range(self.shape[3]):
		#	for j in range(self.shape[2]):
		#		for i in range(self.shape[1]):
		#			fout.write(struct.pack('f',linear[i+j*self.shape[1]+k*self.shape[1]*self.shape[2]]))

		fout.close()

	### Save in binary format, optionally with a molecule info
	def SaveBINmulti(self, filename, mol=None):


		for f in range(self.shape[0]): # loop over fields

			linear = numpy.reshape(self.qube[f], [self.shape[1]*self.shape[2]*self.shape[3]])

			fname = "{}-{}.bin".format(filename, f)
			fout = open(fname,"wb")

			if mol == None:
				fout.write(struct.pack('i',0))
			else:
				fout.write(struct.pack('i',mol.natoms))
				for i in range(mol.natoms):
					fout.write(struct.pack('i',mol.types[i]))
					fout.write(struct.pack('f',mol.coords[i,0]))
					fout.write(struct.pack('f',mol.coords[i,1]))
					fout.write(struct.pack('f',mol.coords[i,2]))

			fout.write(struct.pack('i',self.shape[1]))
			fout.write(struct.pack('i',self.shape[2]))
			fout.write(struct.pack('i',self.shape[3]))

			fout.write(struct.pack('f',self.step))
			fout.write(struct.pack('f',self.origin['x']))
			fout.write(struct.pack('f',self.origin['y']))
			fout.write(struct.pack('f',self.origin['z']))

			for k in range(self.shape[3]):
				for j in range(self.shape[2]):
					for i in range(self.shape[1]):
						fout.write(struct.pack('f',linear[i+j*self.shape[1]+k*self.shape[1]*self.shape[2]]))

			fout.close()



	### Load grid from binary format file
	def LoadBIN(filename):


		fout = open(filename,"rb")

		natm = struct.unpack('i', fout.read(4))[0]
		fout.read(4*4*natm)

		shape = struct.unpack('iii', fout.read(4*3))
		step = struct.unpack('f', fout.read(4))[0]
		origin = struct.unpack('fff', fout.read(4*3))
		
		linear = numpy.zeros((shape[0]*shape[1]*shape[2]), dtype=numpy.float32)
		for k in range(shape[2]):
			for j in range(shape[1]):
				for i in range(shape[0]):
					linear[i+j*shape[0]+k*shape[0]*shape[1]] = struct.unpack('f', fout.read(4))[0]
		fout.close()

		grid = Grid(shape,step)
		grid.step = numpy.float32(step)
		grid.origin = gpuarray.vec.make_float3(origin[0],origin[1],origin[2])

		cuda.memcpy_htod(grid.d_qube, linear)
		cuda.memcpy_dtoh(grid.qube, grid.d_qube)

		return grid



## Enum of possible GP instructions.
@unique
class GPtype(IntEnum):

	CONST = 0
	PROPAGATE = 1
	SCALE = 2
	OFFSET = 3
	ADD = 4
	SUB = 5
	MUL = 6
	DIV = 7
	NN = 8
	TANH = 9
	EXP = 10
	EXP2 = 11


## Represents one genetic program instruction with its parameters.
class GPCode:

	## total number of different instruction types
	nInstructions = len(GPtype.__members__.items())

	## list of instructions that interpret the first argument as an index
	iArg1idx = [GPtype.PROPAGATE, GPtype.SCALE, GPtype.OFFSET, 
		GPtype.ADD, GPtype.SUB, GPtype.MUL, GPtype.DIV, GPtype.TANH, GPtype.EXP, GPtype.EXP2]
	## list of instructions that interpret the second arg as an index
	iArg2idx = [GPtype.ADD, GPtype.SUB, GPtype.MUL, GPtype.DIV]


	## Creates a GP instruction object of given type and arguments
	def __init__(self, typ, args):

		self.type = typ
		self.args = numpy.asarray(args)




	## Returns a random GP instruction.
	def Random(scale):
		
		# const and propagate are not allowed here
		t = numpy.random.randint(2, GPCode.nInstructions)
		
		nargs = 2
		if GPtype(t) == GPtype.NN: nargs = 16+1
		if GPtype(t) == GPtype.CONST: nargs = 1
		args = scale*(2*numpy.random.random(nargs)-1)

		return GPCode(t, args)


	## Converts this instruction into a bytes string.
	def Binarize(self):

		bs = struct.pack('i', int(self.type))
		for a in self.args: bs += struct.pack('f', a)

		return bs


	def LoadBinary(barray):

		ptr = 0
		nargs = 2
		t = struct.unpack("i", barray[0:4])[0]; ptr += 4
		if GPtype(t) == GPtype.NN: nargs = 16+1
		if GPtype(t) == GPtype.CONST: nargs = 1

		args = list(struct.unpack("f"*nargs, barray[ptr:ptr+4*nargs]))

		return GPCode(t, args)


	## Returns the CUDA code for this instruction.
	#
	# inbuf: name of the variable with the input nodes.
	# inbufsize: size of the input nodes buffer.
	# outbuf: name of the variable with the output nodes.
	# outidx: index of the output node for this instruction.
	# indent: indentation (# of tabs).
	def WriteCode(self, inbuf, inbufsize, outbuf, outidx, flipSub=False, constMem=0, indent=1):

		s = "\t"*indent
		if not flipSub:
			s += "{}[{}] = ".format(outbuf, outidx)
		else:
			s += "{}[{}]+= ".format(outbuf, outidx, flipSub)

		t = GPtype(self.type)
		args = list(self.args)

		# convert args to indexes if needed
		if t in GPCode.iArg1idx:
			if numpy.abs(args[0]) < 1: args[0] = int((args[0]*inbufsize) % inbufsize)
			else: args[0] = int(args[0] % inbufsize)
		if t in GPCode.iArg2idx:
			if numpy.abs(args[1]) < 1: args[1] = int((args[1]*inbufsize) % inbufsize)
			else: args[1] = int(args[1] % inbufsize)


		if   t == GPtype.CONST: 	s += "{:4.6f}".format(args[0])

		elif t == GPtype.PROPAGATE: s += "{0}[widx+{1}]".format(inbuf, args[0])
		elif t == GPtype.TANH: 	s += "tanhf({0}[widx+{1}])".format(inbuf, args[0])
		elif t == GPtype.EXP: 	s += "expf(-fabsf({0}[widx+{1}]))".format(inbuf, args[0])
		elif t == GPtype.EXP2: 	s += "expf(-{2} * {0}[widx+{1}]*{0}[widx+{1}])".format(inbuf, args[0], numpy.abs(args[1]))

		elif t == GPtype.SCALE: 	s += "{2:4.6f} * {0}[widx+{1}]".format(inbuf, args[0], args[1])
		elif t == GPtype.OFFSET: 	s += "{2:4.6f} + {0}[widx+{1}]".format(inbuf, args[0], args[1])

		elif t == GPtype.ADD: 	s += "{0}[widx+{1}] + {0}[widx+{2}]".format(inbuf, args[0], args[1])
		elif t == GPtype.SUB: 	s += "{0}[widx+{1}] - {0}[widx+{2}]".format(inbuf, args[0], args[1])
		elif t == GPtype.MUL: 	s += "{0}[widx+{1}] * {0}[widx+{2}]".format(inbuf, args[0], args[1])
		elif t == GPtype.DIV: 	

			snip = "\t"*indent
			snip+= "acc= ({0}[widx+{2}] != 0)? tanhf({0}[widx+{1}] / {0}[widx+{2}]) : 0; ".format(inbuf, args[0], args[1])
			snip+= "if(isnan(acc)) acc=0;\n"
			snip+= "{} acc".format(s)
			s = snip
			#s += "({0}[widx+{2}] != 0)? tanhf({0}[widx+{1}] / {0}[widx+{2}]) : 0".format(inbuf, args[0], args[1],   outbuf,outidx)
		

		#elif t == GPtype.MEAN:
		#	snip = "\t"*indent
		#	snip+= "float acc=0; for(ushort cnt=0; cnt<{}; ++cnt) acc += {}[cnt];\n".format(inbufsize,inbuf)
		#	snip+= s + "acc / {}".format(inbufsize)
		#	s = snip

		elif t == GPtype.NN:
			
			snip = "\t"*indent
			snip+= "acc=0; "
			snip+= "for(ushort iacc=0; iacc<{}; ++iacc) ".format(inbufsize)
			snip+= "acc += cParams[iacc+{1}] * {0}[widx+iacc];\n".format(inbuf, constMem)
			snip += "{}acc = tanhf(acc + cParams[{}]);\n".format("\t"*indent, inbufsize+constMem)
			snip += "{} acc".format(s)
			s = snip

		if flipSub: s += ' * (rep)'
		s += ';\n'
		return s





### Automaton object - GENETIC PROGRAMMING
#
class Automaton:


	def __init__(self):

		self.code = []
		self.nlayers = 0
		self.fitness = 0


	def Random(nlayers, pscale):

		atm = Automaton()
		atm.Randomize(pscale, nlayers)

		return atm



	## Create a random parameter set.
	def Randomize(self, pscale, nlayers):

		self.nlayers = nlayers

		# initial constants
		consts1 = [GPCode(0, pscale*(2*numpy.random.random(1)-1)) for i in range(8)]
		consts2 = [GPCode(0, pscale*(2*numpy.random.random(1)-1)) for i in range(12)]

		# first group is the GP for the q/A/B transfer
		# each layer has 16 instructions,
		# the last layer is the output with only 4 nodes
		self.topo1 = [(16,16) for i in range(nlayers)] + [(16,8), (8,4)]
		ni = 0
		for t in self.topo1: ni += t[1]
		code1 = [GPCode.Random(pscale) for i in range(ni)]

		# second group is GP for A/B creation/destruction
		# 16 node layers and one output layer of size 2
		self.topo2 = [(16,16) for i in range(nlayers)] + [(16,8), (8,4), (4,2)]
		ni = 0
		for t in self.topo2: ni += t[1]
		code2 = [GPCode.Random(pscale) for i in range(ni)]

		self.code = consts1 + code1 + consts2 + code2


	def Initialize(self, cgrid, q0, vne):

		# clear the compute grid
		cgrid.qube *= 0
		cuda.memcpy_htod(cgrid.d_qube, cgrid.qube)

		# copy q0(gpu) to cgrid(gpu)-beginning
		cuda.memcpy_dtod(cgrid.d_qube, q0.d_qube, q0.qube.nbytes)

		# copy vne(gpu) to cgrid(gpu)-offset npts
		ptr = int(cgrid.d_qube) + int(vne.qube.nbytes)
		cuda.memcpy_dtod(ptr, vne.d_qube, vne.qube.nbytes)




	## Writes the compute kernel for the genetic program.
	def WriteCode(self):

		
		src = QMTools.srcAutomaton

		# compute dna size
		dnasize = 0
		for i in self.code: dnasize += len(i.args)
		src = src.replace('PYCUDA_DNASIZE', str(dnasize))
		ic = 0
		ip = 0
		
		flag = '// PYCUDA_GENETICPROGRAM_CONSTS_1'
		codeC = ""
		# set the last 8 inputs to constants
		for i in range(8):
			codeC += "\t\t\tbuffer1[widx+{}] = {:4.6f};\n".format(i+8, self.code[ip].args[0])
			ic += len(self.code[ip].args)
			ip += 1
		src = src.replace(flag, flag+'\n'+codeC)

		# create the code for section 1 - A pass
		nl = 0
		flag = '// PYCUDA_GENETICPROGRAM_1'
		code1 = ""
		for topo in self.topo1:

			inputSize = topo[0]
			outputSize = topo[1]

			for i in range(outputSize):
				c = self.code[ip].WriteCode("buffer1", inputSize, "buffer2", i, flipSub=False, indent=3, constMem=ic)
				code1 += c
				ic += len(self.code[ip].args)
				ip += 1

			if outputSize != 4:
				code1 += "\t\t\tfor(ushort k=0; k<{}; k++) buffer1[widx + k] = buffer2[k];\n\n".format(outputSize)

			# DEBUG! check if some output was nan
			#code1 += "\t\t\tfor(ushort k=0; k<{}; k++) if(isnan(buffer2[k])) printf(\"output here was nan [layer {}, output %i]\\n\",k);\n\n".format(outputSize,nl)
			#nl += 1
		src = src.replace(flag, flag+'\n'+code1)
		# --- END OF PROGRAM 1: transfer rates --- #


		flag = '// PYCUDA_GENETICPROGRAM_CONSTS_2'
		codeC = ""
		# set the last 8 inputs to constants
		for i in range(12):
			codeC += "\tbuffer1[widx+{}] = {:4.6f};\n".format(i+4, self.code[ip].args[0])
			ic += len(self.code[ip].args)
			ip += 1
		src = src.replace(flag, flag+'\n'+codeC)


		# --- PROGRAM TO GENERATE/DESTROY A/B FIELDS --- #
		flag = '// PYCUDA_GENETICPROGRAM_2'
		code2 = ""
		for topo in self.topo2:

			inputSize = topo[0]
			outputSize = topo[1]

			for i in range(outputSize):
				c = self.code[ip].WriteCode("buffer1", inputSize, "buffer2", i, flipSub=False, indent=1, constMem=ic)
				code2 += c
				ic += len(self.code[ip].args)
				ip += 1

			if outputSize != 2:
				code2 += "\tfor(ushort k=0; k<{}; k++) buffer1[widx + k] = buffer2[k];\n\n".format(outputSize)

		src = src.replace(flag, flag+'\n'+code2)
		# --- END OF PROGRAM 2 --- #


		return src


	## Save the automaton parameters in a bytes string.
	def Binarize(self):

		bs = b''

		for i in self.code: bs += i.Binarize()
		
		bs += struct.pack('f', self.fitness)
		
		return bs


	## Load the automaton code from a binary string.
	# The automaton code topology has to be already set
	# returns the offset of the next byte to read from barray
	def LoadBinary(self, barray):

		ptr = 0

		# const1 - 8
		const1 = [GPCode.LoadBinary(barray[i*8:]) for i in range(8)]
		ptr += 8*8
		

		# topo1
		code1 = []
		for t in self.topo1:
			for i in range(t[1]):

				gpc = GPCode.LoadBinary(barray[ptr:])
				code1.append(gpc)
				ptr += 4 + len(gpc.args)*4



		# const2 - 12
		const2 = [GPCode.LoadBinary(barray[ptr+i*8:]) for i in range(12)]
		ptr += 8*12

		# topo2
		code2 = []
		for t in self.topo2:
			for i in range(t[1]):

				gpc = GPCode.LoadBinary(barray[ptr:])
				code2.append(gpc)
				ptr += 4 + len(gpc.args)*4


		self.code = const1 + code1 + const2 + code2

		self.fitness = struct.unpack('f', barray[ptr:ptr+4])[0]
		ptr += 4

		print('automaton loaded [fitness={}]'.format(self.fitness))
		return ptr

	### Compute fitness of this automaton.
	#
	## molecule: reference molecule
	## mol: molcule object
	## cgrid: computing grid, must be already initialised with q0 and vne in channels 0 and 1
	## qref: grid with the correct electron density of the molecule
	def Evolve(self, mol, cgrid, maxiter=1000, tolerance=0.1, debug=False):


		# setup the GP kernel
		kernel = self.WriteCode()

		# common substitutions
		kernel = kernel.replace('PYCUDA_NX', str(cgrid.shape[1]))
		kernel = kernel.replace('PYCUDA_NY', str(cgrid.shape[2]))
		kernel = kernel.replace('PYCUDA_NZ', str(cgrid.shape[3]))
		kernel = kernel.replace('PYCUDA_NPTS', str(cgrid.npts))
		#print(kernel)

		if debug: # print the kernel for debug
			fout = open("atmgp.kernel.txt","w")
			fout.write(kernel)
			fout.close()

		# compile
		ptx = SourceModule(kernel, include_dirs=[os.getcwd()])#, options=["--resource-usage"])

		# get the constant memory pointer
		cp, sb = ptx.get_global('cParams')
		params = []
		for i in self.code: params = params + list(i.args)
		params = numpy.asarray(params).astype(numpy.float32)
		cuda.memcpy_htod(cp, params) # copy constant params


		ptx = ptx.get_function("gpu_automaton_evolve")
		ptx.prepare([numpy.intp, numpy.intp])

		ogrid = Grid.emptyAs(cgrid)

		qtot = QMTools.Compute_qtot(cgrid)

		# debug save
		if debug: # debug save
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
			cgrid.SaveBINmulti('atmgp.input',mol)
			print("qtot at start:", qtot)

		# do evolution
		
		qdiff = tolerance + 1
		rep = 0
		
		while (qdiff > tolerance or rep < 10) and rep < maxiter:

			# propagate q,A,B
			ptx.prepared_call(cgrid.GPUblocks, (8,8,8), cgrid.d_qube, ogrid.d_qube)

			# switch pointers
			tmp = cgrid.d_qube
			cgrid.d_qube = ogrid.d_qube
			ogrid.d_qube = tmp


			# renormalize q if needed
			if rep % 1 == 0:
				qtot = QMTools.Compute_qtot(cgrid)
				#print("step calc {0} -- {1:-5e}".format(rep, qtot))
				if numpy.abs(qtot-mol.qtot) > 1.0e-3:
					factor = float(mol.qtot) / qtot
					#print("rescale",factor)
					QMTools.Compute_qscale(cgrid, factor)
					

			if debug:
				cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
				cgrid.SaveBINmulti('atmgp.output-{}'.format(rep+1),mol)

			# check if q converged
			qdiff = QMTools.Compute_qdiff(cgrid, ogrid, rel=False)
			if numpy.isnan(qdiff):
				print("diff is nan")
				return False, qdiff
			
			#print(rep, qtot, qdiff)

			rep += 1

		return True, qdiff



	def CompareQ(self, cgrid, qref, rel=False):


		qdiff = QMTools.Compute_qdiff(cgrid, qref, rel)
		if numpy.isnan(qdiff):
			qdiff = -9999 + numpy.random.random()
			self.fitness = qdiff
		else: 
			self.fitness = -qdiff

		return self.fitness


	def Mix(a,b):

		r = Automaton()
		r.topo1 = a.topo1
		r.topo2 = a.topo2
		r.nlayers = a.nlayers
		r.code = []

		for ic in range(len(a.code)):
			x = numpy.random.random()
			if x < 0.5: r.code.append(a.code[ic])
			else: r.code.append(b.code[ic])

		return r


	def Mutate(self, amount, mscale, pscale):

		nmuts = int(numpy.ceil(len(self.code) * amount * numpy.random.random()))
		allidx = numpy.arange(len(self.code)).astype(numpy.int32)
		
		idx = numpy.random.choice(allidx, size=nmuts, replace=False)
		idx = idx.astype(numpy.int32)


		for i in idx:

			c = self.code[i]
			if GPtype(c.type) == GPtype.CONST:
				self.code[i].args[0] += mscale*(2*numpy.random.random()-1)
			else:
				# mutation can mutate the instr args or the whole instr
				if numpy.random.random() < 0.2:
					self.code[i] = GPCode.Random(pscale)
				else:

					self.code[i].args += mscale*(2*numpy.random.random(len(self.code[i].args))-1)

### Automaton object - NEURAL NETS
class AutomatonNN:


	def __init__(self):

		self.params = []
		self.absAB = []
		self.nlayers = []
		self.fitness = 0
		self.dnasize = 0
		self.fitness = -9999

	def Random(nl1, nl2, pscale):

		atm = AutomatonNN()
		atm.Randomize(pscale, nl1, nl2)

		return atm



	## Create a random parameter set.
	def Randomize(self, pscale, nl1=4, nl2=2):

		self.nlayers = [nl1, nl2]
		self.dnasize = 16*16*nl1 + 16*nl1 + 16*3 + 3 # first net
		self.dnasize+= 16*16*nl2 + 16*nl2 + 16*2 + 2 # field net
		self.dnasize+= 8 + 12 # extra constants

		self.params = (2*numpy.random.random(self.dnasize)-1)*pscale
		self.params = self.params.astype(numpy.float32)
		self.absAB = numpy.random.random(2)*0.4


	def Initialize(self, cgrid, q0, vne):

		# clear the compute grid
		cgrid.qube *= 0
		cuda.memcpy_htod(cgrid.d_qube, cgrid.qube)

		# copy q0(gpu) to cgrid(gpu)-beginning
		cuda.memcpy_dtod(cgrid.d_qube, q0.d_qube, q0.qube.nbytes)

		# copy vne(gpu) to cgrid(gpu)-offset npts
		ptr = int(cgrid.d_qube) + int(vne.qube.nbytes)
		cuda.memcpy_dtod(ptr, vne.d_qube, vne.qube.nbytes)




	## Writes the compute kernel for the genetic program.
	def WriteCode(self):

		
		src = QMTools.srcAutomatonNN

		src = src.replace('PYCUDA_DNASIZE', str(self.dnasize))
		src = src.replace('PYCUDA_NL1', str(self.nlayers[0]))
		src = src.replace('PYCUDA_NL2', str(self.nlayers[1]))

		src = src.replace('PYCUDA_ABS_A', str(self.absAB[0]))
		src = src.replace('PYCUDA_ABS_B', str(self.absAB[1]))

		return src


	## Save the automaton parameters in a bytes string.
	def Binarize(self):

		bs = b''

		for i in self.params:
			bs += struct.pack('f', i)
		
		bs += struct.pack('f', self.absAB[0])
		bs += struct.pack('f', self.absAB[1])
		
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

		# read abs A/B
		self.absAB = [0,0]
		self.absAB[0] = struct.unpack('f', barray[ptr:ptr+4])[0]; ptr += 4
		self.absAB[1] = struct.unpack('f', barray[ptr:ptr+4])[0]; ptr += 4


		self.fitness = struct.unpack('f', barray[ptr:ptr+4])[0]
		ptr += 4

		print('automaton loaded [fitness={}]'.format(self.fitness))
		return ptr

	### Compute fitness of this automaton.
	#
	## molecule: reference molecule
	## mol: molcule object
	## cgrid: computing grid, must be already initialised with q0 and vne in channels 0 and 1
	## qref: grid with the correct electron density of the molecule
	def Evolve(self, mol, cgrid, maxiter=1000, tolerance=0.1, debug=False):


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
		ptx = SourceModule(kernel, include_dirs=[os.getcwd()]) #, options=["--resource-usage"])

		# get the constant memory pointer
		cp, sb = ptx.get_global('cParams')
		params = numpy.asarray(self.params).astype(numpy.float32)
		cuda.memcpy_htod(cp, params) # copy constant params


		ptx = ptx.get_function("gpu_automaton_nn_evolve")
		ptx.prepare([numpy.intp, numpy.intp])

		ogrid = Grid.emptyAs(cgrid)

		qtot = QMTools.Compute_qtot(cgrid)

		if debug: # debug save
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
			cgrid.SaveBINmulti('atm.input',mol)
			print("qtot at start:", qtot)


		# do evolution
		qdiff = tolerance + 1
		rep = 0

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
				#print(rep, qtot)

			if debug:
				cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)
				cgrid.SaveBINmulti('atm.output-{}'.format(rep+1),mol)


			# check if q converged
			qdiff = QMTools.Compute_qdiff(cgrid, ogrid, rel=False)
			#print(rep, qtot, qdiff)

			if numpy.isnan(qdiff):
				print("diff is nan")
				return False, qdiff
			
			rep += 1

		return True, qdiff



	def CompareQ(self, cgrid, qref, rel=False):


		qdiff = QMTools.Compute_qdiff(cgrid, qref, rel)
		if numpy.isnan(qdiff):
			self.fitness = -9999 + numpy.random.random()
		else: 
			self.fitness = -qdiff

		return self.fitness


	def Mix(a,b):

		r = AutomatonNN()
		r.nlayers = a.nlayers
		r.absAB = [0,0]
		r.params = numpy.copy(a.params)
		r.dnasize = a.dnasize

		for i in range(a.params.shape[0]):

			x = numpy.random.random()
			if x < 0.5: r.params[i] = a.params[i]
			else: r.params[i] = b.params[i]

		for i in range(2):
			x = numpy.random.random()
			if x < 0.5: r.absAB[i] = a.absAB[i]
			else: r.absAB[i] = b.absAB[i]

		return r


	def Mutate(self, amount, mscale, pscale):

		mutagen = (2*numpy.random.random(self.dnasize)-1)*mscale
		mutagen = mutagen.astype(numpy.float32)

		mask = numpy.random.random(self.dnasize).astype(numpy.float32)
		mask[mask>amount] = 0

		self.params += mutagen*mask

		x = numpy.random.random()
		if x < amount:
			x = numpy.random.random()
			if x < 0.5: self.absAB[0] = numpy.random.random()*0.4
			else: self.absAB[0] = numpy.random.random()*0.4


class QMTools:


	fker = open('kernel_utils.cu')
	source = SourceModule(fker.read(), include_dirs=[os.getcwd()], options=["--resource-usage"])
	fker.close()

	kernelQseed = source.get_function("gpu_density_seed")
	kernelQseed.prepare([
		numpy.float32, gpuarray.vec.float3,
		gpuarray.vec.uint3,
		numpy.intp, numpy.intp, numpy.intp
	])

	kernelRescale = source.get_function('gpu_rescale')
	kernelRescale.prepare([ numpy.intp, numpy.float32 ])

	kernelTotal = source.get_function('gpu_total')
	kernelTotal.prepare([ numpy.intp, numpy.intp ])

	kernelQDiff = source.get_function('gpu_qdiff')
	kernelQDiff.prepare([ numpy.intp, numpy.intp, numpy.intp ])

	kernelQDiffRel = source.get_function('gpu_qdiff_rel')
	kernelQDiffRel.prepare([ numpy.intp, numpy.intp, numpy.intp ])

	# computes density grid from density matrix
	fker = open('kernel_density.cu')
	srcDensity = fker.read(); fker.close()

	# computes hartree potential grid from density grid
	fker = open('kernel_hartree.cu')
	srcHartree = fker.read(); fker.close()

	# add nuclear charge to charge grid
	fker = open('kernel_fft.cu')
	srcFFT = fker.read(); fker.close()

	fker = open('kernel_automaton.cu')
	srcAutomaton = fker.read(); fker.close()

	fker = open('kernel_automaton_nn.cu')
	srcAutomatonNN = fker.read(); fker.close()

	h_qtot = numpy.zeros(1).astype(numpy.float32)
	d_qtot = cuda.mem_alloc(sizeofFloat)



	### Compute the electron density on the grid, for a given molecule.
	### function tested and works up to 10^-6!
	### Returns a new grid with the same shape as the input one.
	def Compute_density(grid, molecule, subgrid=1, copyBack=True):

		print("preparing density kernel")
		kernel = QMTools.srcDensity
		kernel = kernel.replace('PYCUDA_NATOMS', str(molecule.natoms))
		kernel = kernel.replace('PYCUDA_NORBS', str(molecule.norbs))

		kernel = kernel.replace('PYCUDA_SUBGRIDN', str(subgrid))
		kernel = kernel.replace('PYCUDA_SUBGRIDDX1', str(1.0/subgrid)+"f")
		kernel = kernel.replace('PYCUDA_SUBGRIDDX2', str(1.0/(subgrid*2))+"f")
		kernel = kernel.replace('PYCUDA_SUBGRIDiV', str(1.0/(subgrid*subgrid*subgrid))+"f")

		#print(kernel)

		kernel = SourceModule(kernel, include_dirs=[os.getcwd()], options=["--resource-usage"])
		kernel = kernel.get_function("gpu_densityqube_shmem_subgrid")
		kernel.prepare([
			numpy.float32, gpuarray.vec.float3,
			numpy.intp,
			numpy.intp, numpy.intp,
			numpy.intp, numpy.intp, numpy.intp
		])

		cgrid = Grid.emptyAs(grid)
		print("computing density qube (no shmem, subgrid {})...".format(subgrid), grid.GPUblocks)

		kernel.prepared_call(grid.GPUblocks, (8,8,8),
			grid.step,
			grid.origin,
			molecule.d_coords,
			molecule.basisset.d_alphas,
			molecule.basisset.d_coeffs,
			molecule.d_ALMOs,
			molecule.d_dm,
			cgrid.d_qube
		)
		
		# compute the total charge
		qtot = QMTools.Compute_qtot(cgrid)

		# rescale
		scaling = molecule.qtot / qtot
		QMTools.Compute_qscale(cgrid, scaling)


		if copyBack:
			cuda.memcpy_dtoh(cgrid.qube, cgrid.d_qube)

		return cgrid


	### Compute the hartree potential on the grid.
	### Returns a new grid with the hartree potential.
	### The output grid has the same specs as the input electron grid.
	def Compute_hartree(vgrid, molecule, copyBack=True):

		print("preparing hartree kernel")
		kernel = QMTools.srcHartree

		# Q grid
		natm = 0
		if molecule != None: natm = molecule.natoms
		kernel = kernel.replace('PYCUDA_NATOMS', str(natm))
		kernel = kernel.replace('PYCUDA_GRID_STEP', str(vgrid.step)+"f")

		kernel = kernel.replace('PYCUDA_GRID_X0', str(vgrid.origin['x'])+"f")
		kernel = kernel.replace('PYCUDA_GRID_Y0', str(vgrid.origin['y'])+"f")
		kernel = kernel.replace('PYCUDA_GRID_Z0', str(vgrid.origin['z'])+"f")

		kernel = kernel.replace('PYCUDA_GRID_NX', str(vgrid.shape[1]))
		kernel = kernel.replace('PYCUDA_GRID_NY', str(vgrid.shape[2]))
		kernel = kernel.replace('PYCUDA_GRID_NZ', str(vgrid.shape[3]))

		kernel = kernel.replace('#define NORB 0', '#define NORB {}'.format(molecule.norbs))

		kernel = SourceModule(kernel, include_dirs=[os.getcwd()], options=["--resource-usage"])
		src = kernel



		#kernel = src.get_function("gpu_hartree_guess")
		#kernel.prepare([numpy.intp, numpy.intp, numpy.intp])
		#kernel.prepared_call(vgrid.GPUblocks, (8,8,8), molecule.d_types, molecule.d_coords, vgrid.d_qube)


		# do jacobi iterations
		kernel = src.get_function("gpu_hartree_GTO")
		kernel.prepare([numpy.intp, numpy.intp, numpy.intp, numpy.intp, numpy.intp, numpy.intp, numpy.intp])
		
		print("computing hartree grid from qube", vgrid.GPUblocks)
			
		kernel.prepared_call(vgrid.GPUblocks, (8,8,8),
			molecule.d_types,
			molecule.d_coords,
			molecule.basisset.d_alphas,
			molecule.basisset.d_coeffs,
			molecule.d_ALMOs,
			molecule.d_dm,
			vgrid.d_qube
		)
		#print("hartree done")


		if copyBack:
			cuda.memcpy_dtoh(vgrid.qube, vgrid.d_qube)
			print("copied")

		return




	### Compute the hartree potential on the grid.
	### Returns a new grid with the hartree potential.
	### The output grid has the same specs as the input electron grid.
	def Compute_hartree_iter(qgrid, molecule, tolerance=1.0e-24, copyBack=True):

		print("preparing hartree kernel")
		kernel = QMTools.srcHartree

		# Q grid
		natm = 0
		if molecule != None: natm = molecule.natoms
		kernel = kernel.replace('PYCUDA_NATOMS', str(natm))
		kernel = kernel.replace('PYCUDA_GRID_STEP', str(qgrid.step)+"f")

		kernel = kernel.replace('PYCUDA_GRID_X0', str(qgrid.origin['x'])+"f")
		kernel = kernel.replace('PYCUDA_GRID_Y0', str(qgrid.origin['y'])+"f")
		kernel = kernel.replace('PYCUDA_GRID_Z0', str(qgrid.origin['z'])+"f")

		kernel = kernel.replace('PYCUDA_GRID_NX', str(qgrid.shape[1]))
		kernel = kernel.replace('PYCUDA_GRID_NY', str(qgrid.shape[2]))
		kernel = kernel.replace('PYCUDA_GRID_NZ', str(qgrid.shape[3]))


		kernel = SourceModule(kernel, include_dirs=[os.getcwd()], options=["--resource-usage"])
		src = kernel


		# place some charges near nuclei
		gqsd = QMTools.Compute_qseed(qgrid, molecule, copyBack=True)
		rho = Grid.emptyAs(qgrid)
		# assuming qgrid is initialised also on the CPU
		rho.qube = gqsd.qube - qgrid.qube
		cuda.memcpy_htod(rho.d_qube, rho.qube)


		# grids for result and computation
		vgrid = Grid.emptyAs(qgrid)
		tgrid = Grid.emptyAs(qgrid)


		# compute initial guess
		tmp = -qgrid.qube*0 / (qgrid.step*qgrid.step*qgrid.step)
		cuda.memcpy_htod(vgrid.d_qube, tmp)

		#kernel = src.get_function("gpu_hartree_guess")
		#kernel.prepare([numpy.intp, numpy.intp, numpy.intp])
		#kernel.prepared_call(vgrid.GPUblocks, (8,8,8), molecule.d_types, molecule.d_coords, vgrid.d_qube)


		# do jacobi iterations
		kernel = src.get_function("gpu_hartree_iteration")
		kernel.prepare([numpy.intp, numpy.intp, numpy.intp])
		
		print("computing hartree grid from qube", vgrid.GPUblocks)
		delta = tolerance+1
		itr = 1
		while delta > tolerance:
		#for i in range(1000):
			
			kernel.prepared_call(vgrid.GPUblocks, (8,8,8), rho.d_qube, vgrid.d_qube, tgrid.d_qube)
			kernel.prepared_call(vgrid.GPUblocks, (8,8,8), rho.d_qube, tgrid.d_qube, vgrid.d_qube)

			if itr % 100 == 0:
				# compare v2 and v1 to see if it converged
				delta = QMTools.Compute_qdiff(vgrid,tgrid)
				print("{0} {1:.8E}".format(itr, delta))

			itr += 1

		'''
		if molecule != None:
			kernel = src.get_function("gpu_hartree_add_nuclei")
			kernel.prepare([numpy.intp, numpy.intp, numpy.intp])
			kernel.prepared_call(vgrid.GPUblocks, (8,8,8), molecule.d_types, molecule.d_coords, vgrid.d_qube)
		'''

		if copyBack:
			cuda.memcpy_dtoh(vgrid.qube, vgrid.d_qube)

		return vgrid


	### Compute the hartree potential on the grid using FFT
	###
	def Compute_hartree_fft(qgrid, molecule, sigma=0.2, cutoff=6):

		print("Preparing FFT kernels")
		kernel = QMTools.srcFFT

		kernel = kernel.replace('PYCUDA_NATOMS', str(molecule.natoms))
		kernel = kernel.replace('PYCUDA_GRID_STEP', str(qgrid.step/ANG2BOR)+"f")

		kernel = kernel.replace('PYCUDA_GRID_X0', str(qgrid.origin['x']/ANG2BOR)+"f")
		kernel = kernel.replace('PYCUDA_GRID_Y0', str(qgrid.origin['y']/ANG2BOR)+"f")
		kernel = kernel.replace('PYCUDA_GRID_Z0', str(qgrid.origin['z']/ANG2BOR)+"f")

		kernel = kernel.replace('PYCUDA_GRID_NX', str(qgrid.space_shape()[0]))
		kernel = kernel.replace('PYCUDA_GRID_NY', str(qgrid.space_shape()[1]))
		kernel = kernel.replace('PYCUDA_GRID_NZ', str(qgrid.space_shape()[2]))

		kernel = kernel.replace('PYCUDA_SIGMA', str(sigma)+"f")
		kernel = kernel.replace('PYCUDA_CUTOFF_SQ', str((cutoff*sigma)**2)+"f")

		kernel = SourceModule(kernel, include_dirs=[os.getcwd()], options=["--resource-usage"])

		kernel_nuclear = kernel.get_function("gpu_add_nuclear_charge")
		kernel_nuclear.prepare([numpy.intp, numpy.intp, numpy.intp, numpy.intp])

		kernel_poisson = kernel.get_function("gpu_poisson_frequency_solve")
		kernel_poisson.prepare([numpy.intp])
		
		print("Adding nuclear charge to qube", qgrid.GPUblocks)

		# Make array to hold total charge density. Has to be of GPUArray type to work with
		# FFT in the next step.
		q_total_qube = gpuarray.GPUArray(qgrid.space_shape(), dtype=numpy.float32)

		kernel_nuclear.prepared_call(qgrid.GPUblocks, (8,8,8),
			qgrid.d_qube,
			q_total_qube.gpudata,
			molecule.d_types,
			molecule.d_coords
		)

		print('Solving the Poisson equation using FFT')

		# Create a temporary complex array for FFT
		cqube = gpuarray.GPUArray(qgrid.space_shape(), dtype=numpy.complex64)

		# Do forward FFT
		q_total_qube = q_total_qube.astype(numpy.complex64)
		plan = cufft.Plan(q_total_qube.shape, numpy.complex64, numpy.complex64)
		cufft.fft(q_total_qube, cqube, plan)

		# Solve Poisson equation in frequency space
		kernel_poisson.prepared_call(qgrid.GPUblocks, (8,8,8),
			cqube.gpudata,
		)

		# Do inverse FFT
		plan_inverse = cufft.Plan(q_total_qube.shape, numpy.complex64, numpy.complex64)
		cufft.ifft(cqube, q_total_qube, plan_inverse, scale=True)
		
		vqube = q_total_qube.real.get()

		return vqube


	### Compute the starting guess for the electron density.
	# The input grid is only a template.
	# returns a new grid with the results
	def Compute_qseed(grid, molecule, copyBack=True):

		print('Computing seed density...')

		# make a single fied grid from this one
		qgrid = Grid.emptyAs(grid, nfields=1)

		# the grid has one block for each atom in the molecule
		# the block is a 2 cube
		shape = grid.space_shape_uint3()

		# since the grid step is small (0.1ang) we can be quite sure
		# that two atoms will not be spread over the same grid points
		# because the molecules are optimised organics!
		QMTools.kernelQseed.prepared_call((molecule.natoms.item(),1,1), (2,2,2),
			qgrid.step,
			qgrid.origin,
			shape,
			molecule.d_types,
			molecule.d_coords,
			qgrid.d_qube
		)
		#cuda.memcpy_dtod(grid.d_qube, qgrid.d_qube, qgrid.qube.nbytes)

		if copyBack:
			cuda.memcpy_dtoh(qgrid.qube, qgrid.d_qube)

		return qgrid


	### Computes the VNe of a molecule and stores it the returned grid
	# The input grid is only used as a template.
	def Compute_VNe(grid, molecule, adsorb=0.1, diff=0.01, tolerance=1.0e-9, copyBack=True):

		print("computing VNe on new grid...")

		# setup the kernel
		fker = open('kernel_vne.cu')
		kernel = fker.read(); fker.close()
		kernel = kernel.replace('PYCUDA_NATOMS',str(molecule.natoms))
		kernel = kernel.replace('PYCUDA_NX',str(grid.shape[1]))
		kernel = kernel.replace('PYCUDA_NY',str(grid.shape[2]))
		kernel = kernel.replace('PYCUDA_NZ',str(grid.shape[3]))
		kernel = kernel.replace('PYCUDA_ADS',str(adsorb))
		kernel = kernel.replace('PYCUDA_DIF',str(diff))
		
		kernel = SourceModule(kernel, include_dirs=[os.getcwd()])
		kernel = kernel.get_function("gpu_vne")
		kernel.prepare([
			numpy.intp,numpy.intp,
			gpuarray.vec.float3,
			numpy.float32,
			numpy.intp, numpy.intp, numpy.intp,
		])

		gridres = Grid.emptyAs(grid, nfields=1)
		gridout = Grid.emptyAs(grid, nfields=1)
		
		h_delta = numpy.zeros(1).astype(numpy.float32)
		h_deltazero = numpy.zeros(1).astype(numpy.float32)
		d_delta = cuda.mem_alloc(h_delta.nbytes); cuda.memcpy_htod(d_delta, h_deltazero)

		h_delta[0] = 1
		niter = 0
		while h_delta[0] > tolerance:
			kernel.prepared_call(grid.GPUblocks, (8,8,8),
				molecule.d_types,
				molecule.d_coords,
				grid.origin,
				grid.step,
				gridres.d_qube,
				gridout.d_qube,
				d_delta
			)
			cuda.memcpy_htod(d_delta, h_deltazero)
			kernel.prepared_call(grid.GPUblocks, (8,8,8),
				molecule.d_types,
				molecule.d_coords,
				grid.origin,
				grid.step,
				gridout.d_qube,
				gridres.d_qube,
				d_delta
			)
			cuda.memcpy_dtoh(h_delta, d_delta)
			cuda.memcpy_htod(d_delta, h_deltazero)
			h_delta /= grid.npts
			
			niter += 1
			if niter > 10*numpy.power(grid.npts,1.0/3):
				print("VNe not converged!")
				break

		if copyBack:
			cuda.memcpy_dtoh(gridres.qube, gridres.d_qube)
		
		return gridres


	### Compute the total charge in a multigrid (first channel)
	def Compute_qtot(grid):

		QMTools.h_qtot[0] = 0
		cuda.memcpy_htod(QMTools.d_qtot, QMTools.h_qtot)
		QMTools.kernelTotal.prepared_call(grid.GPUblocks, (8,8,8), grid.d_qube, QMTools.d_qtot)
		cuda.memcpy_dtoh(QMTools.h_qtot, QMTools.d_qtot)

		return float(QMTools.h_qtot[0])


	### Compute the total charge in a multigrid (first channel)
	def Compute_qscale(grid, factor):

		QMTools.kernelRescale.prepared_call(grid.GPUblocks, (8,8,8), grid.d_qube, factor)
		return



	### Compute the abs diff of two density multigrids (first channel)
	def Compute_qdiff(grid1, grid2, rel=False):

		QMTools.h_qtot[0] = 0
		cuda.memcpy_htod(QMTools.d_qtot, QMTools.h_qtot)

		ker = QMTools.kernelQDiff
		if rel: ker = QMTools.kernelQDiffRel

		ker.prepared_call(grid1.GPUblocks, (8,8,8), grid1.d_qube, grid2.d_qube, QMTools.d_qtot)
		
		cuda.memcpy_dtoh(QMTools.h_qtot, QMTools.d_qtot)
		result = float(QMTools.h_qtot)
		if rel: result *= 100.0

		return result