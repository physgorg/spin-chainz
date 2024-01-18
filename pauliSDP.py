import numpy as np
from time import time
import cvxpy as cp
import matplotlib.pyplot as plt

from exact_XY import *

################################################
# FUNCTIONS FOR SDP SETUP

def pmul(a,b): # lookup multiplication of paulis
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
	return a*b + -1*b*a


# all of these functions below deal with representations of the pOps
def ssum(lst): # "sum" of list of strings
	return ''.join(lst)

def scf(x): # for ___repn___
	if np.real(x)>=0:
		return ' + '+str(x)
	elif np.real(x)<0:
		return ' - '+str(-1*x)

def hhash(pop,N = None): # generate hash from pop object
	return pOp_hash(pop.expr[0],pop.x[0],N=N)

def to_srep(expr,x,n): # inverse of function above (works only for single-term)
	res = ['I']*n
	if len(expr) == 0:
		return ''.join(res)
	for i,y in enumerate(list(expr)):
		res[x[i]] = y
	return ''.join(res)

def spOp(repn): # generate pOp from string repn 'IIIXYIZII...'
	coords = [i for i in range(len(repn)) if repn[i] != 'I']
	expr = repn.replace('I','')
	return pOp(expr,coords)

def ppp(hashval): # convert infinite volume hash to pOp
	if '.' in hashval: # infinite volume hash
		ops = hashval.split('.')
		expr = ''.join([x[0] for x in ops])
		coords = [int(x[1:]) for x in ops]
		return pOp(expr,coords)
	else: # finite volume hash
		coords = [i for i in range(len(hashval)) if hashval[i] != 'I']
		expr = hashval.replace('I','')
		return pOp(expr,coords)
	
def cyclic_hash(s): # cyclic invariant hash function IMPORTANT FUNCTION 
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

def pOp_hash(expr,x,N = None): # general hash function: turns pOp into a referencecable hash value. 
	if N == None: # then we are in the thermo limit
		if len(expr) == 0:
			return 'I'
		zero = x[0]
		s = ''
		for i in range(len(expr)):
			if i == 0:
				s += expr[i]+str(x[i] - zero) # this makes the hash automatically 1-periodic
			else:
				s += '.'+expr[i]+str(x[i] - zero)
		return s
	else:
		srep = to_srep(expr,x,N)
		return cyclic_hash(srep)
		

def pOp_normalOrder(new_expr,coords): # multiply out a matmul string of paulis
	product = ''
	new_coords = []
	new_coeff = 1
	while len(new_expr) > 0:
		end = 1
		try:
			while end < len(coords) and coords[end] == coords[0]:
				end += 1
		except IndexError:
			end = 1
		cluster = new_expr[:end]
		while len(cluster)>1:
			cf,op = pmul(cluster[0],cluster[1])
			new_coeff *= cf
			if len(cluster) == 2:
				cluster = op
			else:
				cluster = op + cluster[2:]

		if cluster[0] != 'I':
			product += cluster[0]
			new_coords.append(coords[0])

		coords = coords[end:]
		new_expr = new_expr[end:]

	return product,new_coords,new_coeff

#################################################
# FUNCTIONS FOR ANALYSIS OF SDP OUTPUT (PHYSICS)


def getCritExp(x,y,N = None,cutoff = 3):
	x = x[cutoff:-cutoff]
	y = y[cutoff:-cutoff]
	if N == None: # fit to CFT continuum expression
		reg = linregress(np.log(np.array(x)),np.log(np.array(y)))
		return reg.slope*-1/2
	else: # fit to cylinder CFT expression
		reg = linregress(np.log(np.sin(np.pi*np.array(x)/N)**2),np.log(np.abs(np.array(y))))
		return reg.slope*-1

def Ising_corrCompare(prob,gam=1,mu=1,title = None): # compare correlation functions to exact (finite-size) Ising solution
	corrs = prob.get_corrs()
	N = prob.N

	rv, XX = corrs['XX']
	rv, YY = corrs['YY']
	rv, ZZ = corrs['ZZ']
	XXc = XX
	YYc = YY
	ZZc = ZZ - corrs['Z']**2

	exact_corrs = Ising_corrs(mu,gam,N)
	rv, iXX = exact_corrs['XX']
	rv, iYY = exact_corrs['YY']
	rv, iZZ = exact_corrs['ZZ']
	iXXc = iXX
	iYYc = iYY
	iZZc = iZZ 

	sdp_exp = getCritExp(rv,XX,N)
	exact_exp = getCritExp(rv,iXX,N)

	print("XX critical exponent (SDP):",sdp_exp)
	print("XX critical exponent (exact):",exact_exp)

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
	if title != None:
		fig.suptitle(title)
	plt.show()

##################################################
# CLASSES

class pOp: # pauli operators (pstring equivalent)

	def __init__(self,expr,x,coeffs = None):

		# Deal with inputs; initialize everything in same format
		if isinstance(expr,str): # if expr is string, we are specifying a product of ops
			self.n = 1 # number of operators in sum
			if len(expr) == 1 and str(x).isnumeric(): # if expr is only one, it's a single-site operator X(0)
				coord_arr = [[x]]
				expr = [expr]             
			else: # it is a product of multiple sites
				coord_arr = [x]
				expr = [expr]
		elif isinstance(expr,list): # if expr is list, we are specifying a sum of products of ops
			coord_arr = x

		if len(expr) != len(coord_arr):
			raise ValueError("Operator and coordinate inputs must be the same size. ")
		if coeffs == None:
			coeffs = np.ones(len(coord_arr))

		# sort ops by length, sort each op, multiply terms, combine like terms
		term_exprs = []
		unfiltered_coeffs = []
		lengths = [len(y) for y in expr]
		length_order = np.argsort(lengths)
		
		for i in length_order:
			ops = expr[i]
			nums = coord_arr[i]
			
			inds = np.argsort(nums)
			nums.sort()
			ops = ssum([ops[j] for j in inds]) # order by coordinate

			ops,nums,cf = pOp_normalOrder(ops,nums) # multiply through stuff 
			cf = cf*coeffs[i]
			
			term = [ops,nums]
			if term in term_exprs:
				loc = term_exprs.index(term)
				unfiltered_coeffs[loc] += cf
			else:
				term_exprs.append(term)
				unfiltered_coeffs.append(cf)
			

		self.expr = [x[0] for i,x in enumerate(term_exprs) if unfiltered_coeffs[i] != 0]
		self.x = [x[1] for i,x in enumerate(term_exprs) if unfiltered_coeffs[i] != 0]
		self.cfs = [x for x in unfiltered_coeffs if x != 0]
		self.n = len(self.expr)

		# define name
		self.names = []
		if len(self.expr) == 0:
			self.names.append('0')
		for i in range(len(self.expr)):
			coeff = self.cfs[i]
			op = self.expr[i]
			coord = self.x[i]
			subname = ssum([op[j] + '({})'.format(coord[j]) for j in range(len(op))])
			self.names.append(subname)           

	def __repr__(self):
		if len(self.expr) == 0:
			return '0'
		name = ''
		for i in range(len(self.expr)):
			subname = self.names[i]
			coeff = self.cfs[i]
			if i == 0:
				name += str(coeff)+subname
			else:
				name += scf(coeff)+subname  
		return name

	# override addition
	def __add__(self,other):
		if isinstance(other,pOp):
			return pOp(self.expr + other.expr,self.x + other.x,self.cfs + other.cfs)
		elif isinstance(other,float) or isinstance(other,complex) or isinstance(other,int):
			return pOp(self.expr + ['I'],self.x + [[0]],self.cfs + [other])
	def __radd__(self,other):
		return self.__add__(other)
	def __sub__(self,other):
		return self.__add__(-1*other)

	def __mul__(self,other):
		if isinstance(other,float) or isinstance(other,complex) or isinstance(other,int):
			return pOp(self.expr,self.x,[other*cf for cf in self.cfs])
		elif isinstance(other,pOp):
			new_exprs = []
			new_coords = []
			new_cfs = []
			for i in range(self.n): # for each member of left part,
				Lop = self.expr[i]
				Lcoord = self.x[i]
				Lcf = self.cfs[i]
				for j in range(other.n): # multiply each element of right part (new object will simplify it)
					Rop = other.expr[j]
					Rcoord = other.x[j]
					Rcf = other.cfs[j]
					new_exprs.append(Lop + Rop)
					new_coords.append(Lcoord + Rcoord)
					new_cfs.append(Lcf*Rcf) 
			return pOp(new_exprs,new_coords,new_cfs)

	def __rmul__(self,other):
		if isinstance(other,float) or isinstance(other,complex) or isinstance(other,int):
			return pOp(self.expr,self.x,[other*cf for cf in self.cfs])
		
	def translate(self,d,N = None):
		newcoords = []
		for coords in self.x:
			if N == None:
				newcoords.append([y + d for y in coords])
			else:
				newcoords.append([(y+d) % N for y in coords])
		return pOp(self.expr,newcoords,coeffs = self.cfs)
		
	def asList(self):
		res = []
		for i in range(self.n):
			res.append(pOp(self.expr[i],self.x[i])*self.cfs[i])
		return res

class pauliSDP: # spin chain SDP class

	def __init__(self,basis_ops,Ham = None,N = None,v = False,anchored = False,timed = False):

		self.v = v
		
		self.timed = timed
		
		self.basis = [pOp('I',0)]
		
		self.N = N
		
		# add hamiltonian
		if Ham != None:
			self.full_ham,self.hashd = self.H(Ham) 
		
		if N != None: # finite volume case
			self.idhash = cyclic_hash(ssum(['I']*N))
			if anchored: # orbit all one-point funcs, keep others anchored
				for op in basis_ops: 
					if len(op.expr) == 1 and len(op.expr[0]) == 1:
						self.basis += [pOp(op.expr[0],n) for n in range(N)]
					else:
						self.basis.append(op)
			else: # orbit all operators
				for op in basis_ops:
					opname = op.expr[0] # assume only multiplicative ops
					opx = op.x[0]
					self.basis += [pOp(opname,[(z+n) % N for z in opx]) for n in range(N)]
		else: # infinite volume case
			self.idhash = 'I'
			self.basis += basis_ops
			
		self.Mshape = (len(self.basis),len(self.basis))

		if v: print("Operator basis:", self.basis)
	
		# Create the positivty matrix
		# Since the objects have algebra all multiplication/commute/anticommute is done here
		t = time()
		self.duals = []
		self.dual_hashes = []
		self.Fmats = {}
		for i in range(len(self.basis)):
			for j in range(i,len(self.basis)):
				
				elem = self.basis[i]*self.basis[j] # main multiplication step here

				for k in range(elem.n): # elem = 2X(n)Z(m)
					coeff = elem.cfs[k] # 2
					op = elem.expr[k] # 'XZ'
					xv = elem.x[k] # [n,m]
					hash_val = pOp_hash(op,xv,N = N) # 'X0.Zm-n.' 'U,P,V,X' "U0.P1.V3..."
					if hash_val not in self.Fmats.keys(): # if we have not got this variable yet
						if hash_val != self.idhash:
							self.duals.append(pOp(op,xv))
							self.dual_hashes.append(hash_val)
						self.Fmats[hash_val] = np.zeros(self.Mshape,dtype = complex)
						self.Fmats[hash_val][i,j] = coeff
						self.Fmats[hash_val][j,i] = np.conj(coeff)
					elif hash_val in self.Fmats.keys():
						self.Fmats[hash_val][i,j] = coeff
						self.Fmats[hash_val][j,i] = np.conj(coeff)
		  
		self.commed = [com(self.full_ham,op) for op in self.basis]
	
		# Create second positivity matrix
#         self.Gmats = {val:np.zeros(self.Mshape,dtype = complex) for val in self.dual_hashes}
#         self.Gmats[self.idhash] = np.zeros(self.Mshape,dtype = complex)
#         for i in range(len(self.basis)):
#             for j in range(i,len(self.basis)):

#                 elem = self.basis[i]*self.commed[j] # main multiplication step here

#                 for k in range(elem.n):
#                     coeff = elem.cfs[k]
#                     op = elem.expr[k]
#                     xv = elem.x[k]
#                     hash_val = pOp_hash(op,xv,N = N)
#                     if hash_val not in self.Gmats.keys(): # if we have not got this variable yet
#                         if hash_val != self.idhash:
#                             self.duals.append(pOp(op,xv))
#                             self.dual_hashes.append(hash_val)
#                         self.Gmats[hash_val] = np.zeros(self.Mshape,dtype = complex)
#                         self.Gmats[hash_val][i,j] = coeff
#                         self.Gmats[hash_val][j,i] = np.conj(coeff)
#                     elif hash_val in self.Gmats.keys():
#                         self.Gmats[hash_val][i,j] = coeff
#                         self.Gmats[hash_val][j,i] = np.conj(coeff)
					
		tt = time()     
		if timed: print('find M, F/Gmats time =',tt -t)

		self.n_duals = len(self.duals) # identity does not count as a dual
		
		if v: print("Number of dual variables:",self.n_duals)
			
		self.c = np.zeros((self.n_duals)) # construct objective function-to-be
		for i,hsh in enumerate(self.dual_hashes):
			try:
				self.c[i] += self.hashd[hsh]
			except KeyError:
				continue

		self.solved = False

		self.result = None
		self.opt_dict = None
	  

	def H(self,h): # add the Hamiltonian to the problem
		if self.N == None: # infinite volume; only need to consider the energy density
			hashd = {pOp_hash(h.expr[i],h.x[i]):h.cfs[i] for i in range(h.n)}
		else: # finite volume; 
			# construct full Hamiltonian for whole length of system
			h = sum([h.translate(n,N = self.N) for n in range(self.N)])
			hashd = {pOp_hash(h.expr[i],h.x[i],N = self.N):h.cfs[i] for i in range(h.n)}     
		self.full_ham,self.hashd = [h,hashd]
		return [h,hashd]
	
	def gen_cnstr(self,Ax,rhs=0,real = False):
		if real:
			v = np.zeros(self.n_duals)
		else:
			v = np.zeros(self.n_duals,dtype = complex)
		for i in range(Ax.n):
			hashval = pOp_hash(Ax.expr[i],Ax.x[i],N = self.N)
			coeff = Ax.cfs[i]
			if hashval == self.idhash:
				rhs -= coeff
			else:
				try:
					ind = self.dual_hashes.index(hashval)
					v[ind] = coeff
				except ValueError:
					print()
					print("Operator",pOp(Ax.expr[i],Ax.x[i]), "wasn't found in basis.")
		return v,rhs

	def solve(self,extras = True):

		t = time()
		xvar = cp.Variable(self.n_duals) # define vector of dual variables

		Fm = self.Fmats # matrices F_i which comprise M = F_0 + F_1*x1 + ...

		M = Fm[self.idhash] + sum([Fm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals) if self.dual_hashes[i] in Fm.keys()])

		constraints = [M>>0] # standard semidefinite constraint
		
#         Gm = self.Gmats
		
#         D = Gm[self.idhash] + sum([Gm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals) if self.dual_hashes[i] in Gm.keys()])
		
#         constraints += [D>>0]
		
		c = self.c
		objective = cp.Minimize(c.T @ xvar) # set objective function to minimize the energy

		if extras:
			redun1 = [] # keep track of constraints to avoid redundancy
			redun2 = []
			for op in self.basis[1:]:
				commed = com(self.full_ham,op) # add <[H,O]> = 0 constraints
				ax,b = self.gen_cnstr(commed)
				lax = ax.tolist()
				if lax not in redun1:
					constraints += [ax.T @ xvar == b] 
					redun1.append(lax)

				mult = op*commed # add <O[H,O]> >= 0 constraints
				mult = pOp(mult.expr,mult.x,coeffs = [np.real(cf) for cf in mult.cfs])
				ax,b = self.gen_cnstr(mult,real = True)
				lax = ax.tolist()
				if lax not in redun2:
					redun2.append(lax)

					constraints += [ax.T @ xvar >= b] # add <O[H,O]> >= 0 constraints
			
		sdp = cp.Problem(objective,constraints)

		result = sdp.solve(solver = cp.MOSEK,verbose = False)
		optimal = xvar.value
		
		self.opt_dict = {self.dual_hashes[i]:optimal[i] for i in range(self.n_duals)}

		self.result = (result,optimal)
		if self.v: print("Result:",result)
		self.solved = True

		tt = time()
		if self.timed: print('solve time',tt-t)

		return result,optimal
	
	def get_corrs(self):
		if self.solved == False:
			print("Problem is not solved.")
			return None
	
		corrdict = {}
		
		onepts = ['X','Y','Z']
		for op in onepts:
			pop = pOp(op,0)
			hashval = hhash(pop,self.N)
			try:
				optval = self.opt_dict[hashval]
			except KeyError:
				print("Correlator",op,'not present in matrix.')
				continue
			if abs(optval) < 1e-10:
				optval = 0
			corrdict[op] = optval
		
		twopts = ['XX','YY','ZZ']
		for op in twopts:
			sites = np.arange(self.N+1)
			optvals = np.zeros(self.N+1)
			optvals[0] = optvals[-1] = 1
			for i in range(1,self.N):
				pop = pOp(op,[0,i])
				hashval = hhash(pop,self.N)
				try:
					optval = self.opt_dict[hashval]
				except KeyError:
					print("Correlator",op,'not present in matrix.')
					continue
				optvals[i] = optval
			data = np.asarray([sites,optvals])
			corrdict[op] = data
			
		return corrdict
		
	
	
	
	def range_slackSolve(self,erange,warm_start = True):

		t0 = time()
		
		xvar = cp.Variable(self.n_duals) # correlators
		
		t = cp.Variable() # slack variable

		Fm = self.Fmats

		M = Fm[self.idhash] + sum([Fm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals) if self.dual_hashes[i] in Fm.keys()]) - t*np.identity(self.Mshape[0])
		constraints = [M>>0] # M - tId >> 0
		
		c = self.c
		e = cp.Parameter() # energy parameter
		
		constraints += [c.T @ xvar == e] # fix the energy
		
		objective = cp.Maximize(t)
		


		results = []

		
		for energy in tqdm(erange):
			
			e.value = energy
			HH = self.full_ham
			
			# add nonlinear energy constraints here (For now, only use 1pt funcs)
			nonlin_constraints = []
			nonlin_constraints += constraints # can't modify in-place here
			
			for op in self.basis:
				
				# add <[H,O]> = 0 constraints
				comm = op*HH - HH*op
				ax,b = self.gen_cnstr(comm)
				nonlin_constraints += [ax.T @ xvar == b]
				
				# add <HO> = E<O> constraints
				shifted = HH - energy
				ax,b = self.gen_cnstr(shifted*op)
				nonlin_constraints += [ax.T @ xvar == b]

			sdp = cp.Problem(objective,nonlin_constraints)
			result = sdp.solve(solver = cp.MOSEK,verbose = False,warm_start = warm_start)
			optimal = xvar.value
			results.append((result,optimal))
#                 optimals.append(optimal)

		t1 = time()
		if self.timed: print('solve time',t1-t0)

		return results

if __name__ == '__main__':
	
	xx = spOp('XX')
	z = spOp('Z')
	x = spOp('X')
	y = spOp('Y')

	h = -1*xx - z 

	basis = ['X','Y','Z','XX','XY','ZZ']
	basis = [ppp(x) for x in basis]

	Nsites = 12

	prob = pauliSDP(basis,Ham = h,N = Nsites,v = False,anchored = True,timed = True)

	res,opt = prob.solve(extras = True)

	exact = XY_GS(1,1,Nsites)/Nsites

	print("Exact:",exact)
	print("SDP:",res)
	print("Percent error:",abs((exact-res)/exact)*100)





