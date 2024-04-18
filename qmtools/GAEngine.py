import builtins
import struct

import numpy


class GAElement:

	"""
		Dummy element object for GAEngine operation.
		Any derived class should override these methods with custom code:
			1. Random
			2. Serialize/Deseriaize
			3. Mix
			4. Mutate
			5. Evaluate

	"""


	def __init__(self):

		self.fitness = -9999
		self.params = []
		
		self._dnasize = 0

		self._precision = 8
		self._format = 'd'



	def Random(**kwargs):

		pscale = kwargs.get('pscale', 1.0)

		e = GAElement()
		e._dnasize = 5
		e.params = pscale * (2*numpy.random.random(e._dnasize)-1)

		return e


	## Returns a bytes string with the element serialization.
	def Serialize(self):

		bs = b''

		for i in self.params:
			bs += struct.pack(self._format, i)
		
		bs += struct.pack(self._format, self.fitness)

		return bs

	### Load the element from byte array, returns number of bytes read.
	def Deserialize(self, barray):

		ptr = 0
		
		# read the big params
		for i in range(self.params.shape[0]):
			self.params[i] = struct.unpack(self._format, barray[ptr:ptr+self._precision])[0]
			ptr += self._precision

		self.fitness = struct.unpack(self._format, barray[ptr:ptr+self._precision])[0]
		ptr += self._precision

		print('GAElement loaded [fitness={}]'.format(self.fitness))
		return ptr


	def Mix(a,b):

		r = a.__class__()
		r.params = numpy.copy(a.params)
		r._dnasize = a._dnasize


		mask = numpy.random.random(a.params.shape[0])
		maska = r.params * 1
		maskb = r.params * 1
		
		maska[mask>=0.5] = 0
		maska[mask<0.5] = 1
		
		maskb[maska==1] = 0
		maskb[maska==0] = 1

		r.params = maska * a.params + maskb * b.params

		return r

	
	def Mutate(self, amount, mscale, **kwargs):

		dt = self.params.dtype

		mutagen = (2*numpy.random.random(self._dnasize)-1)*mscale
		mutagen = mutagen.astype(dt)

		mask = numpy.random.random(self._dnasize).astype(dt)
		mask[mask<=amount] = 1
		mask[mask>amount] = 0

		self.params += mutagen*mask


	def Evaluate(self, **kwargs):


		refdata = kwargs.get('refdata', None)
		if refdata is None:
			print('invalid refdata!')
			return 0

		xs = refdata[:,0]
		ys = refdata[:,1]

		yp = ys * 0
		for i in range(self.params.shape[0]):
			
			yp += self.params[i]*numpy.sin(xs*(i+1))

		error = numpy.mean(numpy.abs(yp-ys))

		self.fitness = -error
		return self.fitness

	



class GA:

	PARAMS = {
		'float': 	['mutationRate', 'mutationSize', 'pscale'],
		'int': 		['populationSize', 'rerands', 'rstfreq'],
		'bool':		['keepbest'],
	}

	def ProcessParameter(name, value):


		for k in GA.PARAMS.keys():

			lst = GA.PARAMS[k]
			func = vars(builtins)[k]

			if name in lst:
				return func(value)


		return value

	def ReadParameters(plist):

		opts = {}
		for i in range(len(plist)):

			if not plist[i].startswith('--'): continue

			pname = plist[i][2:]
			value = GA.ProcessParameter(pname, plist[i+1])

			opts[pname] = value

		return opts



	def __init__(self, **kwargs):


		self.populationSize = kwargs.get('populationSize', 128)
		self.keepbest = kwargs.get('keepbest', False)
		self.rerands = kwargs.get('rerands', 1)

		self.mutationRate = kwargs.get('mutationRate', 0.02)
		self.mutationSize = kwargs.get('mutationSize', 0.10)

		self.pscale = kwargs.get('pscale', 1.0)

		self.logfile = kwargs.get('logfile', "ga.log")
		self.outfile = kwargs.get('outfile', "ga.out.bin")
		self.rstfile = kwargs.get('rstfile', "ga.restart.bin")
		self.rstfreq = kwargs.get('rstfreq', 1)
		self.restart = kwargs.get('restart', None)
		

		self.indexes = numpy.arange(self.populationSize)

		self.elementTemplate = None
		self.elementInitDict = {}


		self.population = None
		self.offspring = None
		self.fits = []

		self.generation = 0



	'''
		GA needs to know:
			1. the class to create for the elements
			2. how to initialize them
			3. how to evaluate them
			4. how to mix them
	'''


	def Initialize(self, noLog=False):

		initer = dict(self.elementInitDict)
		initer['pscale'] = self.pscale

		self.offspring = [self.elementTemplate.Random(**initer) for i in range(self.populationSize)]
		
		if not noLog:
			self.flog = open(self.logfile, "w")

		if self.restart is not None:
			
			frst = open(self.restart, "rb")
			barray = frst.read()

			for i in range(self.populationSize):
				ptr = self.offspring[i].Deserialize(barray)
				barray = barray[ptr:]

			frst.close()

			# the population was written after evaluation, then we should make an offspring one

			self.population = self.offspring
			self.fits = numpy.asarray([e.fitness for e in self.population])
			
			# create the new population by mixing the loaded one
			self.Reproduce()



		return


	def Evolve(self, **kwargs):

		"""
		Perform one iteration of GA.
		"""

		debug = kwargs.get('debug', False)


		self.population = [self.offspring[i] for i in range(self.populationSize)]
		self.fits = numpy.zeros(self.populationSize)

		# evaluate the whole population
		for i in range(self.populationSize):

			atm = self.population[i]
			
			self.fits[i] = atm.Evaluate(**kwargs)

			if debug:
				print("generation[{}] element[{}] fitness {}".format(self.generation, i, self.fits[i]))


		fitmax = numpy.max(self.fits)
		fitmin = numpy.min(self.fits)
		fitmean = numpy.mean(self.fits)
		print("generation {0}: best {1:8.5f} -- worst {2:8.5f} -- mean {3:8.5f}".format(self.generation, fitmax, fitmin, fitmean))


		# make a restart
		if self.generation % 1 == 0:

			frst = open(self.rstfile, "wb")
			for i in range(self.populationSize):
				frst.write(self.population[i].Serialize())
			frst.close()
		
		self.generation += 1

		# print the best one to a file
		maxfitidx = numpy.argmax(self.fits)
		self.flog.write("{} {} {}\n".format(fitmax, fitmin, fitmean))
		self.flog.flush()

		# mix the new population
		self.Reproduce()

		return


	def Reproduce(self):

		"""
		Create a new population from the current one by mixing the elements.
		The population has to be already evaluated.

		"""

		# the population has to be sorted, or the softer selection method will not work
		asrt = numpy.argsort(-self.fits) # fits are negative, and argsort assumes ascending order

		# now fits will be resorted so that the best element is first
		self.fits = self.fits[asrt]
		
		# this is the population in the sorted order
		pop = []
		for i in asrt: pop.append(self.population[i])


		fitmax 		= numpy.max(self.fits)
		fitmin		= numpy.min(self.fits)
		fitmean 	= numpy.mean(self.fits)
		maxfitidx 	= numpy.argmax(self.fits)

		indexes = numpy.arange(self.populationSize)


		# generate offspring
		# this uses a softer approach
		k = numpy.log(10.0)/len(pop)
		prob = numpy.exp(-k * numpy.arange(len(pop)))
		prob /= numpy.sum(prob)
		

		p1 = numpy.random.choice(indexes, replace=True, size=self.populationSize, p=prob)
		p2 = p1 * 0
		for i in range(self.populationSize):
			otheridx = numpy.select([indexes != p1[i]],[indexes])
			p2[i] = numpy.random.choice(otheridx, p=prob)

		pairs = numpy.transpose(numpy.asarray([p1,p2]))
		
		# mix
		self.offspring = [ self.elementTemplate.Mix(pop[p[0]], pop[p[1]]) for p in pairs ]
		
		# mutate
		for atm in self.offspring:
			if numpy.random.random() < self.mutationRate: atm.Mutate(self.mutationSize, self.pscale)
		
		# keep the best one
		if self.keepbest: self.offspring[0] = pop[maxfitidx]

		# random immigrants - rerands
		initer = dict(self.elementInitDict)
		initer['pscale'] = self.pscale
		for i in range(self.rerands): 
			self.offspring[self.populationSize - 1 - i] = self.elementTemplate.Random(**initer)


		return








