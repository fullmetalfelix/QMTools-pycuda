import numpy
from numpy.ctypeslib import ndpointer
from ctypes import *
import pickle
import struct
import os
from enum import IntEnum, unique

ANG2BOR = 1.8897259886

MAXAOS = 15
MAXAOC = 20
MOLFAT = 2.0

sizeofInt = 4
sizeofFloat = 4
sizeofDouble = 8


class float3(Structure):

	_fields_ = [
		('x', c_float),
		('y', c_float),
		('z', c_float),
	]


class molecule(Structure):
	_fields_ = [
		('natm', 	c_int),
		('Zs', 		POINTER(c_int)),
		('xyz', 	POINTER(c_float)),

		('norb', 	c_int),
		('dm', 		POINTER(c_float)),
		('almos', 	POINTER(c_short)),

		('alphas', 	POINTER(c_float)),
		('coeffs', 	POINTER(c_float)),
	]

molecule_p = POINTER(molecule)



dblib = cdll.LoadLibrary("./libdatabaser.so")
c_short_p = POINTER(c_short)

NSHELLS = c_int.in_dll(dblib, "nshells")
NSHELLS.value = 6

SHELLREPS = c_int.in_dll(dblib, "shellreps")
SHELLREPS.value = 5

dblib.compute_output.argtypes = [molecule_p, c_float, c_int, c_uint, 
	ndpointer(c_float, flags="C_CONTIGUOUS")
]
dblib.compute_output.restype = c_int


#dblib.compute_density.argtypes = [molecule_p, float3, c_float, c_int, POINTER(c_float), POINTER(c_float)]
#dblib.compute_density.restype = None




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

		print("basis set read - maxorbs={}  ncmax={} - nparams={}\n".format(maxorbs, ncmax, offset))


## Stores a molecule in BOHR, also on the GPU.
class Molecule:


	def __init__(self, filexyz, filedm, basisset):

		self.CID = 0
		self.basisset = basisset

		m = None
		if isinstance(filexyz, list): m = numpy.asarray(filexyz)
		else:
			print("opening molecule (ang):", filexyz)
			m = numpy.loadtxt(filexyz)
			if len(m.shape) == 1:
				m = numpy.asarray([m])
		
		# make coordinates in bohr
		m[:,1:] *= ANG2BOR
		self.types  = numpy.array(m[:,0], dtype=numpy.int32)
		self.coords = numpy.array(m[:,1:], dtype=numpy.float32)
		self.natoms = numpy.int32(self.types.shape[0])
		# total nuclear charge
		self.qtot = numpy.int32((numpy.sum(self.types)))


		# load the DM
		dm = None
		if isinstance(filedm, numpy.ndarray): dm = filedm
		else:

			print("opening density matrix:", filedm)
			dm = numpy.load(filedm)

		self.dm = numpy.array(dm, dtype=numpy.float32)
		self.norbs = numpy.int32(dm.shape[0])
		
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



	def as_molecule_struct(self):

		m = molecule()
		
		m.natm = self.natoms
		m.Zs = self.types.ctypes.data_as(POINTER(c_int))
		m.xyz = self.coords.ctypes.data_as(POINTER(c_float))

		m.norb = self.norbs
		m.dm = self.dm.ctypes.data_as(POINTER(c_float))
		m.almos = self.ALMOs.ctypes.data_as(POINTER(c_short))


		m.alphas = self.basisset.alphas.ctypes.data_as(POINTER(c_float))
		m.coeffs = self.basisset.coeffs.ctypes.data_as(POINTER(c_float))

		return m


	# end of Molecule class





class DensityCalculator:

	
	


	def Compute(mol, basis):


		m = mol.as_molecule_struct()

		nevals = mol.natoms * (NSHELLS.value*SHELLREPS.value + 1) # on and around atoms
		nevals+= SHELLREPS.value # far away shell
		nevals+= mol.natoms * mol.natoms * SHELLREPS.value # between bonds




		#print("molecule {} # points {}".format(mol.CID, nevals))
		output = numpy.zeros((nevals,4), dtype=numpy.float32)



		ee = dblib.compute_output(byref(m), c_float(0.1*ANG2BOR), c_int(2), c_uint(mol.CID), output)
		output = output[0:ee]

		#print(output)
		#print(ee,nevals);

		return output


