import numpy as np
from time import time
import cvxpy as cp
import matplotlib.pyplot as plt

from exact_XY import *

################################################
# FUNCTIONS FOR SDP SETUP

def pmul(a,b): # lookup multiplication of pstrings
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

def ssum(lst): # "sum" of list of strings
	return ''.join(lst)

def scf(x):
	if np.real(x)>=0:
		return ' + '+str(x)
	elif np.real(x)<0:
		return ' - '+str(-1*x)

def hhash(pop):
	return pOp_hash(pop.expr[0],pop.x[0])

def topop(repn):
	coords = [i for i in range(len(repn)) if repn[i] != 'I']
	expr = repn.replace('I','')
	return pOp(expr,coords)


def ppp(hashval):
	ops = hashval.split('.')
	expr = ''.join([x[0] for x in ops])
	coords = [int(x[1:]) for x in ops]
	return pOp(expr,coords)

def pOp_hash(expr,x):
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


def getCorrelators(problem):
    if not problem.solved: # solve problem if not already
        problem.solve()
    energy,xvec = problem.result # optimal values
    ops = np.array(problem.dual_hashes) # all ops w/ unique value
#     print(ops)
    # short_ops = np.array([x.replace('I','') for x in ops])
    # sym_corrs = list(np.unique(short_ops)) # find which types of correlators we have
    corr_dict = {}
    for i,op in enumerate(ops): # for each type of correlator,

        # inds = np.where((short_ops == corr) | (short_ops == corr[::-1]))#
#         print(corr_dict)
        bits = op.split('.')
#         print(bits)
        v = [x[0] for x in bits]
        key = ssum(v)
#         print(op,xvec[i])
        c = [int(x[1:]) for x in bits]
        # print('c',c)
#         print(key)
        if len(v) == 1: # if a one-point func,
            corr_dict[key] = [xvec[i]] # just put the value        

        elif len(v) == 2 and v[0] == v[1]: # if a two-point func,
            if key not in corr_dict.keys():
#                 print('new key',key)
                
                corr_dict[key] = {abs(c[1]-c[0]):xvec[i]}
            else:
                corr_dict[key][abs(c[1]-c[0])] = xvec[i]
#                 print('c',c)
#                 corr_dict[key][1].append()
        elif len(v) == 3:
            continue
#     print(corr_dict.keys())
    for key in corr_dict.keys():
        if len(key) == 2:
#             print(key)
            subdict = corr_dict[key]
            vals = np.sort(list(subdict.keys()))
            arr = [subdict[val] for val in vals]
            corr_dict[key] = (vals,arr)
        
    return corr_dict


##################################################
# CLASSES

class pOp: # pauli operators (pstring equivalent)

	def __init__(self,expr,x,coeffs = None):

		# Initialization steps: 
		#  - cast everything as expr = ['O1O2','O3'], x = [[0,1],[1]]
		#  - sort each product of ops by coordinate (not commuting ops at same coordinate)
		#  - do operator multiplication in each product
		
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
			self.n = len(expr)
			coord_arr = [x[i] for i in range(self.n)]

		if len(expr) != len(coord_arr):
			raise ValueError("Operator and coordinate inputs must be the same size. ")
			
		lengths = [len(y) for y in expr]
		length_order = np.argsort(lengths)
		if coeffs == None:
			coeffs = np.ones(self.n)
		
		# sort ops by length, sort each op, multiply terms, combine like terms
		self.expr = []
		self.x = []
		self.cfs = []
		for i in length_order:
			ops = expr[i]
			nums = coord_arr[i]
			inds = np.argsort(nums)
			nums.sort()
			ops = ssum([ops[j] for j in inds])
			
			ops,nums,cf = pOp_normalOrder(ops,nums) # multiply through stuff 
			cf = cf*coeffs[i]
			
			if ops in self.expr and nums in self.x: # add linear combos
				loc = self.expr.index(ops)
				self.cfs[loc] += cf # add coefffs
			else: # otherwise append
				self.expr.append(ops)
				self.x.append(nums)
				self.cfs.append(cf)
				
		# process for zero coeffs
		self.expr = [x for i,x in enumerate(self.expr) if self.cfs[i] != 0]
		self.x = [x for i,x in enumerate(self.x) if self.cfs[i] != 0]
		self.cfs = [x for x in self.cfs if x != 0]
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
	#     def __rsub__(self,other):
	#         return -1*self
		
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

class oo_pauliSDP: # spin chain SDP class

	def __init__(self,basis_ops,Ham = None,v = False,anchored = False,timed = False):

		self.timed = timed

		self.basis = [pOp('I',0)] + basis_ops # for now, always include identity
	
		self.Mshape = (len(self.basis),len(self.basis))

		if v: print("Operator basis:"); print(self.basis)

		# Since the objects have algebra all multiplication/commute/anticommute is done here
		t = time()
		self.duals = []
		self.dual_hashes = []
		self.Fmats = {}
		for i in range(len(self.basis)):
			for j in range(i,len(self.basis)):
				
				elem = self.basis[i]*self.basis[j] # main multiplication step here
				
				for k in range(elem.n):
					coeff = elem.cfs[k]
					op = elem.expr[k]
					xv = elem.x[k]
					hash_val = pOp_hash(op,xv)
					if hash_val not in self.Fmats.keys(): # if we have not got this variable yet
						if hash_val != 'I':
							self.duals.append(elem)
							self.dual_hashes.append(hash_val)
						self.Fmats[hash_val] = np.zeros(self.Mshape,dtype = complex)
						self.Fmats[hash_val][i,j] = coeff
						self.Fmats[hash_val][j,i] = np.conj(coeff)
					elif hash_val in self.Fmats.keys():
						self.Fmats[hash_val][i,j] = coeff
						self.Fmats[hash_val][j,i] = np.conj(coeff)
		tt = time()     
		if timed: print('find M, Fmats time =',tt -t)

		self.n_duals = len(self.duals) # identity does not count as a dual

		self.Hcom_ops = []

		self.solved = False
		self.v = v

		self.result = None

		self.c = np.zeros((self.n_duals)) # objective function-to-be

		if Ham != None:
			self.full_ham = self.H(Ham)

	def H(self,h):
		t = time()
		hashd = {pOp_hash(h.expr[i],h.x[i]):h.cfs[i] for i in range(h.n)}
		for i,hsh in enumerate(self.dual_hashes):
			try:
				self.c[i] += hashd[hsh]
			except KeyError:
				continue
		tt = time()
		if self.timed: print("ham time",tt-t)
		return [h,hashd]

	#     def add_Hcom(self,op):
	#         if isinstance(op,list):
	#             for x in op:
	#                 self.add_Hcom(x)
	#         elif isinstance(op,str):
	#             self.Hcom_ops.append(pstr(op))


	def solve(self):
		xvar = cp.Variable(self.n_duals)

		Fm = self.Fmats

		M = Fm['I'] + sum([Fm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals)])
		constraints = [M>>0]

		t = time()
			# add commutator constraints
	#         Hops,hash_dict = self.full_ham
	#         for op in self.Hcom_ops:
	#             op = pstr(Ifill(op.expr,self.N))
	#             first_H_op = pstr(Hops[0],coeff = hash_dict[cyclic_hash(Hops[0])])
	#             coms = com(first_H_op,op)
	#             for x in Hops[1:]:
	#                 hashval = cyclic_hash(x)
	#                 Hop = pstr(x,coeff = hash_dict[hashval])
	#                 commutate= com(Hop,op)
	#                 coms+= commutate
	#             cnstr_cfs = coms.cfs
	#             try:
	#                 cnstr_vars = [self.dual_hashes.index(cyclic_hash(x)) for x in coms.ops]
	#                 constraints += [sum([xvar[cnstr_vars[i]]*cnstr_cfs[i] for i in range(len(cnstr_vars))]) == 0]
	#             except ValueError:
	#                 continue

		tt = time()
		if self.timed: print("Hcom op time",tt-t)
		t = time()

		c = self.c
		objective = cp.Minimize(c.T @ xvar)

		sdp = cp.Problem(objective,constraints)

		result = sdp.solve(solver = cp.MOSEK,verbose = False)
		optimal = xvar.value

		self.result = (result,optimal)
		if self.v: print("Result:",result)
		self.solved = True

		tt = time()
		if self.timed: print('solve time',tt-t)

		return result,optimal

	def fixedE_solve(self,energy,obj_c,sense = 'min'):
		xvar = cp.Variable(self.n_duals)


		Fm = self.Fmats

		M = Fm['I'] + sum([Fm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals)])
		constraints = [M>>0]

		t = time()

		tt = time()
		if self.timed: print("Hcom op time",tt-t)
		t = time()

		# fix the energy
		constraints += [(self.c.T @ xvar) == energy]

		if sense == 'min':
			objective = cp.Minimize(obj_c.T @ xvar)
		elif sense == 'max':
			objective = cp.Maximize(obj_c.T @ xvar)

		sdp = cp.Problem(objective,constraints)

		result = sdp.solve(verbose = False)
		optimal = xvar.value

		self.result = (result,optimal)
		if self.v: print("Result:",result)
		self.solved = True

		tt = time()
		if self.timed: print('solve time',tt-t)

		return result,optimal

	def slackSolve(self,energy):

		xvar = cp.Variable(self.n_duals)

		t = cp.Variable()

		Fm = self.Fmats

		M = Fm['I'] + sum([Fm[self.dual_hashes[i]]*xvar[i] for i in range(self.n_duals)])
		iden = np.identity(M.shape[0])
		constraints = [(M-t*iden)>>0]

		t = time()

		tt = time()
		if self.timed: print("Hcom op time",tt-t)
		t = time()

		# fix the energy
		constraints += [(self.c.T @ xvar) == energy]

		objective = cp.Maximize(t)
			# objective = cp.Minimize(obj_c.T @ xvar)
		
			# objective = cp.Maximize(obj_c.T @ xvar)

		sdp = cp.Problem(objective,constraints)

		result = sdp.solve(verbose = False)
		optimal = xvar.value

		self.result = (result,optimal)
		if self.v: print("Result:",result)
		self.solved = True

		tt = time()
		if self.timed: print('solve time',tt-t)

		return result,optimal



if __name__ == '__main__':

	xx = pOp('XY',[0,0])
	# xx = pAlg([xx])
	print(xx)
	yy = pOp('XY',[0,0])
	# yy = pAlg([yy])
	print(yy)
	prod = xx -1*yy + 2

	print(prod)














