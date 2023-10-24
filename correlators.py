import numpy as np
import matplotlib.pyplot as plt
from time import time
from pauli_SDP_lib import *

from scipy.stats import linregress

######################################

# CORRELATORS FROM SDP OBJECT

def v2str(corr):
	l = ['x','y','z']
	res = []
	for j,x in enumerate(corr):
		res += [l[j] for i in range(x)]
	return ssum(res)

def getCorrelators(problem):
	N = problem.N
	if not problem.solved: # solve problem if not already
		problem.solve()
	energy,xvec = problem.result # optimal values
	orbits = problem.orbits # dual variables, symbolically
	ops = np.array([x[0] for x in orbits]) # all ops w/ unique value
	
	short_ops = np.array([x.expr.replace('I','') for x in ops])
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
			prod = pstring(o1)*pstring(o2)
			if prod - pstring("I") == 0: # if they are idem,
				vals[0] = 1
				vals[N] = 1
			else: # otherwise they multiply to something we already have
				vals[0] = None
				vals[N] = None
				
			dists = [op.expr.index(o2)-op.expr.rfind(o1) for op in v]
			d1 = [d % N for d in dists] # find distances mod N
			d2 = [abs(d) for d in dists]
			vals[d1] = x
			vals[d2] = x

			if corr[::-1] not in list(corr_dict.keys()):
				corr_dict[corr]= (dr,vals)
		else:
			corr_dict[corr] = (x,v)
		
	return corr_dict
		
def getCritExp(x,y,N = None):
    if N == None:
        reg = linregress(np.log(np.array(x)),np.log(np.array(y)))
        return reg.slope*-1/2
    else:
        reg = linregress(np.log(np.sin(np.pi*np.array(x)/N)**2),np.log(np.abs(np.array(y))))
        return reg.slope*-1














if __name__ == '__main__':
	
	zz = pstring('ZZ')
	x = pstring('X')

	mu = 1

	h = -1*zz - mu*x

	N = 8

	t = time()
	yz = pSDP(['X','Y','Z'],N,Ham = h)
	yz_res,_ = yz.solve()
	tt = time()

	print("yz time",tt-t)
	print('yz res',yz_res)

	t = time()
	corrs = getCorrelators(yz)
	tt = time()
	print('corr time',tt-t)
	print("CORRELATORS")
	for x in corrs:
		print(x,corrs[x])

	