from enum import IntEnum, auto, unique
import numpy


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


class GPCode:


	nInstructions = len(GPtype.__members__.items())
	iArg1idx = [GPtype.PROPAGATE, GPtype.SCALE, GPtype.OFFSET, 
		GPtype.ADD, GPtype.SUB, GPtype.MUL, GPtype.DIV, GPtype.TANH, GPtype.EXP]
	iArg2idx = [GPtype.ADD, GPtype.SUB, GPtype.MUL, GPtype.DIV]


	def __init__(self, typ, args):

		self.type = typ
		self.args = args



	def Random(scale):
		
		t = numpy.random.randint(0, GPCode.nInstructions)
		
		nargs = 2
		if GPtype(t) == GPtype.NN: nargs = 16+1
		args = scale*(2*numpy.random.random(nargs)-1)

		return GPCode(t, args)




	def WriteCode(self, inbuf, inbufsize, outbuf, outidx, indent=1):

		s = "\t"*indent
		s+= "{}[{}] = ".format(outbuf, outidx)

		t = GPtype(self.type)
		args = list(self.args)

		if t in GPCode.iArg1idx:  args[0] = int(args[0] % inbufsize)
		if t in GPCode.iArg2idx:  args[1] = int(args[1] % inbufsize)


		if   t == GPtype.CONST: 	s += "{:4.6f}".format(args[0])
		elif t == GPtype.PROPAGATE: s += "{0}[{1}]".format(inbuf, args[0])
		elif t == GPtype.SCALE: 	s += "{2:4.6f} * {0}[{1}]".format(inbuf, args[0], args[1])
		elif t == GPtype.OFFSET: 	s += "{2:4.6f} + {0}[{1}]".format(inbuf, args[0], args[1])
		elif t == GPtype.ADD: 	s += "{0}[{1}] + {0}[{2}]".format(inbuf, args[0], args[1])
		elif t == GPtype.SUB: 	s += "{0}[{1}] - {0}[{2}]".format(inbuf, args[0], args[1])
		elif t == GPtype.MUL: 	s += "{0}[{1}] * {0}[{2}]".format(inbuf, args[0], args[1])
		elif t == GPtype.DIV: 	s += "({0}[{2}] != 0)? {0}[{1}] / {0}[{2}] : 0".format(inbuf, args[0], args[1])
		elif t == GPtype.TANH: 	s += "tanhf({0}[{1}])".format(inbuf, args[0])
		elif t == GPtype.EXP: 	s += "expf(-absf({0}[{1}]))".format(inbuf, args[0])

		#elif t == GPtype.MEAN:
		#	snip = "\t"*indent
		#	snip+= "float acc=0; for(ushort cnt=0; cnt<{}; ++cnt) acc += {}[cnt];\n".format(inbufsize,inbuf)
		#	snip+= s + "acc / {}".format(inbufsize)
		#	s = snip

		elif t == GPtype.NN:
			
			snip = "\t"*indent
			snip+= "acc=0;\n"
			for i in range(inbufsize):
				tmp = "\t"*indent
				tmp += "acc += {2:4.6f} * {0}[{1}];\n".format(inbuf, i, args[i])
				snip += tmp
			snip += "{}acc = tanhf(acc + {:4.6f});\n".format("\t"*indent, args[inbufsize])
			snip += "{} acc".format(s)
			s = snip

		s += ';\n'
		return s



for rep in range(10):

	i = GPCode.Random(10)
	c = i.WriteCode('in', 8, 'output',0,0)
	print(c)

