import numpy as np
from time import time
import cvxpy as cp
import matplotlib.pyplot as plt

from exact_XY import *

################################################
# FUNCTIONS

def pmul(a,b):
	results = {
		'XY' : [1j,'Z'],
		'YX' : [-1j,'Z'],
		'XZ' : [-1j,'Y'],
		'ZX' : [1j,'Y'],
		'YZ' : [1j,'X'],
		'ZY' : [-1j,'X']}
	if a == "I":
		return [1,b]
	elif b == 'I':
		return [1,a]
	elif a == b:
		return [1,'I']
	else:
		return results[a+b]

def com(a,b):
	return a*b + -1.0*b*a

def cyclic_hash(s):
	doubled_s = s + s
	rotations = [doubled_s[i:i+len(s)] for i in range(len(s))]
	smallest_rotation = min(rotations)
	
	base = 7
	mod = 10**9 + 7
	hash_value = 0
	for char in smallest_rotation:
		# ord(char) converts the character to its ASCII value
		hash_value = (hash_value * base + ord(char)) % mod
	return hash_value

def Ilst(n): 
	return ['I' for i in range(n)]

def ssum(lst): # "sum" of list of strings
	return ''.join(lst)

def Ifill(s,n):
	missing = ssum(Ilst(n-len(s)))
	return s+missing

def sroll(s,n):
    return s[-n:] + s[:-n]

def psimplify(cfs,exprs):
	ops = []
	coeffs = []
	for op in np.unique(exprs):
		locs = np.where(exprs == op)[0]
		coeff = sum(cfs[locs])
		if coeff == 0:
			continue
		else:
			coeffs.append(coeff)
			ops.append(op)
	return coeffs,ops

def constructBasis(oplist,n_sites):
	ops = [x for x in oplist if x != 'I']
	N = n_sites
	operators = []
	if 'I' in oplist:
		withIdentity = True
	else:
		withIdentity = False
	L = len(ops)
	for j in range(N): # site index
		for i in range(L): # operator index
			basis_op = Ilst(N) # initialized 
			ssite_op = list(ops[i])
			oplen = len(ssite_op)
			for x in range(oplen):
				basis_op[(j+x) % N] = ssite_op[x]
			operators.append(ssum(basis_op))
	print(operators)
	operators = [pstr(x) for x in operators]
	if withIdentity: 
		return [pstr(ssum(Ilst(N)))] + operators
	else:
		return operators

def constructAnchorBasis(oplist,n_sites):
	ops = [x for x in oplist if x != 'I']
	N = n_sites
	operators = []
	if 'I' in oplist:
		withIdentity = True
	else:
		withIdentity = False
	L = len(ops)
	for i in range(L): # operator index
		ssite_op = list(ops[i])
		oplen = len(ssite_op)
		if oplen <= 1: # don't anchor 1pt operators in basis
			for j in range(N): # site index
				basis_op = Ilst(N)
				for x in range(oplen):
					basis_op[(j+x) % N] = ssite_op[x]
				operators.append(ssum(basis_op))
		else: # only include these at a specific site
			basis_op = Ilst(N)
			for x in range(oplen):
				basis_op[x % N] = ssite_op[x]
			operators.append(ssum(basis_op))
	operators = [pstr(x) for x in operators]
	if withIdentity: 
		return [pstr(ssum(Ilst(N)))] + operators
	else:
		return operators

def getCorrelators(problem):
	N = problem.N
	if not problem.solved: # solve problem if not already
		problem.solve()
	energy,xvec = problem.result # optimal values
	ops = np.array(problem.duals) # all ops w/ unique value
	
	short_ops = np.array([x.replace('I','') for x in ops])
	sym_corrs = list(np.unique(short_ops)) # find which types of correlators we have
	corr_dict = {}
	for corr in sym_corrs: # for each type of correlator,
		
		inds = np.where((short_ops == corr) | (short_ops == corr[::-1]))#

		v = ops[inds] # get corresponding pstrings
		x = xvec[inds] # get optimal values

		if len(corr) == 1: # if a one-point func,
			corr_dict[corr] = [xvec[inds][0]] # just put the value
			
		elif len(corr) == 2: # if a two-point func,
			dr = np.arange(N+1)
			vals = np.zeros((N+1))
			o1,o2 = corr[0],corr[1]
									
			# compute the onsite product explicitly
			prod = pstr(o1)*pstr(o2)
			if prod.expr == 'I': # if they are idem,
				vals[0] = 1
				vals[N] = 1
			else: # otherwise they multiply to something we already have
				vals[0] = None
				vals[N] = None
				
			dists = [op.index(o2)-op.rfind(o1) for op in v]
			d1 = [d % N for d in dists] # find distances mod N
			d2 = [abs(d) for d in dists]
			vals[d1] = x
			vals[d2] = x

			if corr[::-1] not in list(corr_dict.keys()):
				corr_dict[corr]= (dr,vals)
		else:
			corr_dict[corr] = (x,v)
		
	return corr_dict

def getCritExp(x,y,N = None,cutoff = 1):
	x = x[cutoff:-cutoff]
	y = y[cutoff:-cutoff]
	if N == None:
		reg = linregress(np.log(np.array(x)),np.log(np.array(y)))
		return reg.slope*-1/2
	else:
		reg = linregress(np.log(np.sin(np.pi*np.array(x)/N)**2),np.log(np.abs(np.array(y))))
		return reg.slope*-1


def Ising_corrCompare(prob,mu):
	corrs = getCorrelators(prob)
	N = prob.N

	rv, XX = corrs['XX']
	rv, YY = corrs['YY']
	rv, ZZ = corrs['ZZ']
	XXc = XX
	YYc = YY
	ZZc = ZZ - corrs['Z'][0]**2

	exact_corrs = Ising_corrs(mu,N)
	rv, iXX = exact_corrs['XX']
	rv, iYY = exact_corrs['YY']
	rv, iZZ = exact_corrs['ZZ']
	iXXc = iXX
	iYYc = iYY
	iZZc = iZZ 

	fig,ax = plt.subplots(1,3,figsize = (15,5))
	x2,y2,z2 = ax

	x2.plot(rv,XX,label = 'SDP')
	x2.plot(rv,iXXc,'k--',label = 'exact')
	x2.legend()
	x2.set_title(r"Correlator $\langle XX \rangle$; N = {}, $\mu = {}$".format(N,mu))
	x2.set_ylabel(r"value of corr. func")
	x2.set_xlabel(r"distance")

	y2.plot(rv,YY,label = 'SDP')
	y2.plot(rv,iYYc,'k--',label = 'exact')
	y2.legend()
	y2.set_title(r"Correlator $\langle YY \rangle$; N = {}, $\mu = {}$".format(N,mu))
	# y2.set_ylabel(r"value of corr. func")
	y2.set_xlabel(r"distance")

	z2.plot(rv,ZZc,label = 'SDP')
	z2.plot(rv,iZZc,'k--',label = 'exact')
	z2.legend()
	z2.set_title(r"Correlator $\langle ZZ \rangle$; N = {}, $\mu = {}$".format(N,mu))
	# z2.set_ylabel(r"value of corr. func")
	z2.set_xlabel(r"distance")
	plt.show()

##################################################
# CLASSES

class pstr:
	
	def __init__(self,expr,coeff = 1):
		
		self.expr = expr
		self.cf = coeff
		
	def __repr__(self):
		if self.cf == 1:
			return self.expr
		else:
			return str(self.cf)+self.expr
		
	def __mul__(self,other):
		if isinstance(other,float) or isinstance(other,complex) or isinstance(other,int):
			return pstr(self.expr,coeff = self.cf*other)
		elif isinstance(other,pstr):
			if len(self.expr) == len(other.expr): # for same-length strings,
				coeff = self.cf*other.cf
				s1 = list(self.expr)
				s2 = list(other.expr)
				prod = [pmul(s1[i],s2[i]) for i in range(len(self.expr))]
				coeff = coeff *np.prod([x[0] for x in prod])
				s3 = ''.join([x[1] for x in prod])
				return pstr(s3,coeff = coeff)   
		
	def __rmul__(self,other):
		if isinstance(other,float) or isinstance(other,complex) or isinstance(other,int):
			return pstr(self.expr,coeff = self.cf*other)

	def __add__(self,other):
		if isinstance(other,pstr):
			return pVec([self,other])
		if isinstance(other,pVec):
			return pVec([self]+other.args)

	def __radd__(self,other):
		return self.__add__(other)

	def __sub__(self,other):
		if isinstance(other,pstr):
			return pVec([self,-1*other])

	def __rsub__(self,other):
		if isinstance(other,pstr):
			return pVec([other,-1*pstr])

class pVec:
	
	def __init__(self,pstrs):
		
		raw_coeffs = np.array([x.cf for x in pstrs])
		raw_ops = np.array([x.expr for x in pstrs])
		# simplify step
		self.cfs,self.ops = psimplify(raw_coeffs,raw_ops)

		self.args = [pstr(self.ops[i],coeff = self.cfs[i]) for i in range(len(self.cfs))]

	def __repr__(self):
		if len(self.ops) == 0:
			return '0'
		srep = ''
		def scf(x):
			if np.real(x)>=0:
				return ' + '+str(x)
			elif np.real(x)<0:
				return ' - '+str(-1*x)

		for i in range(len(self.cfs)):
			coeff = self.cfs[i]
			op = self.ops[i]
			if i == 0:
				srep += str(coeff)+op
			else:
				srep += scf(coeff)+op
		return srep

	def __add__(self,other):

		if isinstance(other,pstr):
			return pVec(self.args + [other])

		if isinstance(other,pVec):
			return pVec(self.args + other.args)

	def __radd__(self,other):
		return self.__add__(other)

	def __sub__(self,other):
		return self.__add__(-1*other)

	def __rsub__(self,other):
		return -1*self.__add__(-1*other)

	def __mul__(self,other):
		if str(other).isnumeric():
			return pVec([other*x for x in self.args])
	def __rmul__(self,other):
		if str(other).isnumeric():
			return self.__mul__(other)





class pauliSDP: # spin chain SDP class
	
	def __init__(self,basis_ops,N,Ham = None,v = False,anchored = False):
		
		self.N = N # number of sites

		t = time()
		if not anchored:
			self.basis = constructBasis(basis_ops,N)
		else:
			self.basis = constructAnchorBasis(basis_ops,N)
		tt = time()
		# print('constructBasis time =',tt-t)
		
		self.withIdentity = ('I' in basis_ops)
		if self.withIdentity:
			self.L = len(basis_ops) - 1
		else:
			self.L = len(basis_ops)
			
		self.Mshape = (len(self.basis),len(self.basis))
		
		self.idhash = cyclic_hash(ssum(Ilst(N)))
		
		if v: print("Operator basis:"); print(self.basis)
		
		# Since the objects have algebra all multiplication/commute/anticommute is done here
		t = time()
		self.duals = []
		self.dual_hashes = []
		self.Fmats = {}
		for i in range(len(self.basis)):
			for j in range(i,len(self.basis)):
				elem = self.basis[i]*self.basis[j]
				coeff = elem.cf
				s = elem.expr
				hash_val = cyclic_hash(s)
				if hash_val not in self.Fmats.keys(): # if we have not got this variable yet
					if hash_val != self.idhash:
						self.duals.append(s)
						self.dual_hashes.append(hash_val)
					self.Fmats[hash_val] = np.zeros(self.Mshape,dtype = complex)
					self.Fmats[hash_val][i,j] = coeff
					self.Fmats[hash_val][j,i] = np.conj(coeff)
				elif hash_val in self.Fmats.keys():
					self.Fmats[hash_val][i,j] = coeff
					self.Fmats[hash_val][j,i] = np.conj(coeff)
		tt = time()     
		print('find M, Fmats time =',tt -t)
		
		self.n_duals = len(self.duals) # identity does not count as a dual
		
		self.Hcom_ops = []

		self.solved = False

		self.result = None
		
		self.c = np.zeros((self.n_duals)) # objective function-to-be

		if Ham != None:
			self.full_ham = self.H(Ham)
	
	def H(self,h):
		t = time()
		# construct vars for whole length of system
		full_ham_vars = [var.expr + ssum(Ilst(self.N-len(var.expr))) for var in h.args]
		# compute hashes
		ham_hashes = [cyclic_hash(x) for x in full_ham_vars]
		cfs = [var.cf for var in h.args]

		hashd = {ham_hashes[i]:cfs[i] for i in range(len(ham_hashes))}
		full_ham = []
		for x in full_ham_vars:
			for n in range(self.N):
				full_ham.append(sroll(x,n))

		for i in range(len(full_ham_vars)):
			ind = self.dual_hashes.index(ham_hashes[i])
			self.c[ind] = cfs[i]
		tt = time()
		print("ham time",tt-t)
		return [full_ham,hashd]

	def add_Hcom(self,op):
		if isinstance(op,list):
			for x in op:
				self.add_Hcom(x)
		elif isinstance(op,str):
			self.Hcom_ops.append(pstr(op))

	
	def solve(self):
		xvar = cp.Variable(self.n_duals)

		Fm = self.Fmats
		
		M = Fm[self.idhash] + sum([Fm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals)])
		constraints = [M>>0]

		t = time()
		# add commutator constraints
		Hops,hash_dict = self.full_ham
		for op in self.Hcom_ops:
			op = pstr(Ifill(op.expr,self.N))
			first_H_op = pstr(Hops[0],coeff = hash_dict[cyclic_hash(Hops[0])])
			coms = com(first_H_op,op)
			for x in Hops[1:]:
				hashval = cyclic_hash(x)
				Hop = pstr(x,coeff = hash_dict[hashval])
				commutate= com(Hop,op)
				coms+= commutate
			cnstr_cfs = coms.cfs
			try:
				cnstr_vars = [self.dual_hashes.index(cyclic_hash(x)) for x in coms.ops]
				constraints += [sum([xvar[cnstr_vars[i]]*cnstr_cfs[i] for i in range(len(cnstr_vars))]) == 0]
			except ValueError:
				continue

		tt = time()
		print("Hcom op time",tt-t)
		t = time()
		
		c = self.c
		objective = cp.Minimize(c.T @ xvar)

		sdp = cp.Problem(objective,constraints)

		result = sdp.solve(solver = cp.MOSEK,verbose = False)
		optimal = xvar.value
		# rescale by size of orbit if needed
#         optimal = np.array([self.N/(len(self.orbits[i]))*x for i,x in enumerate(optimal)])

		self.result = (result,optimal)
		self.solved = True

		tt = time()
		print('solve time',tt-t)

		return result,optimal




if __name__ == '__main__':
	
	N = 10


	xx = pstr('XX')
	z = pstr('Z')
	bas = ['I','X','Y','Z','XX','YY','ZZ']

	t1 = time()
	prob = pauliSDP(bas,N,Ham = -1*xx - z,anchored = True)

	prob.add_Hcom(bas[1:])
	# prob.add_Hcom('XX')
	# prob.add_Hcom('YY')
	# prob.add_Hcom('X')
	# prob.add_Hcom('Y')
	# prob.add_Hcom('Z')
	# prob.add_Hcom('XY')

	# prob.constraints

	res,opt = prob.solve()

	
	t2 = time()
	print("result:",res)
	print("total time:",t2-t1)




