import numpy as np
import matplotlib.pyplot as plt
from time import time
from pauli_SDP_lib import *
from exact_XY import *

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










if __name__ == '__main__':
	
	xx = pstring('XX')
	z = pstring('Z')

	mu = 1

	h = -1*xx - mu*z

	N = 8
	t = time()
	prob = pSDP(['I','X','Y','Z','XY','XZ'],N,Ham = h)

	prob.add_Hcom_cnstr('X')
	prob.add_Hcom_cnstr('Y')
	prob.add_Hcom_cnstr('Z')

	res,opt = prob.solve()
	tt = time()

	print("time",tt-t)
	print('SDP result',res)

	exact = XY_GS(mu,1,N)



	print('exact:',exact)

	print('percent error:',100*abs((exact - res)/exact))

	Ising_corrCompare(prob,mu)

	