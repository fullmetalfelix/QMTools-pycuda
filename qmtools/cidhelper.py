import base64
import io

import numpy
import requests

requests.packages.urllib3.disable_warnings()


### access to database
class CIDHelper:

	SERVER = 'https://69.12.4.38/CCSD'
	CIDs = []


	def ProcessResponse(rsp):

		if rsp.status_code != 200:
			print('unable to connect to server')
			return None

		answer = rsp.json()
		if answer['type'] == 'error':
			print('error response')
			return None

		return answer

	def GetCIDs():


		req = requests.get('{}/CIDs'.format(CIDHelper.SERVER), verify=False)
		answer = CIDHelper.ProcessResponse(req)
		if answer is None: return None

		CIDHelper.CIDs = answer['CIDs']


	def Init():
		CIDHelper.GetCIDs()


	def GetMolecule(CID):

		if CID not in CIDHelper.CIDs:
			print('molecule {} not available'.format(CID))
			return None


		req = requests.get('{}/dccsd/{}'.format(CIDHelper.SERVER, CID), verify=False)
		answer = CIDHelper.ProcessResponse(req)
		if answer is None: return None


		#answer['xyz'] = numpy.array(answer['xyz'])

		b = answer['D-CCSD'].get('$binary',{}).get('base64',None)
		if b is None:
			print("invalid molecule data")
			return None

		b = base64.b64decode(b)
		bio = io.BytesIO(b)

		matrix = numpy.load(bio)
		
		answer['CID'] = CID
		answer['D-CCSD'] = matrix
		return answer

	def GetRandomMolecule():

		cid = numpy.random.choice(CIDHelper.CIDs)
		return CIDHelper.GetMolecule(cid)





CIDHelper.Init()
