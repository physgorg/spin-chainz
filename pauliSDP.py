import numpy as np
from time import time
import cvxpy as cp
import matplotlib.pyplot as plt

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
    operators = [pstr(x) for x in operators]
    if withIdentity: 
        return [pstr(ssum(Ilst(N)))] + operators
    else:
        return operators

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


class pauliSDP: # spin chain SDP class
    
    def __init__(self,basis_ops,N,Ham = None,v = False):
        
        self.N = N # number of sites

        t = time()
        self.basis = constructBasis(basis_ops,N)
        tt = time()
        print('constructBasis time =',tt-t)
        
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
            for j in range(len(self.basis)):
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
                elif hash_val in self.Fmats.keys():
                    self.Fmats[hash_val][i,j] = coeff
        tt = time()     
        print('find M, Fmats time =',tt -t)
        
        self.n_duals = len(self.duals) # identity does not count as a dual
        
        self.Hcom_ops = []

        self.solved = False

        self.result = None
        
        self.c = np.zeros((self.n_duals)) # objective function-to-be

        if Ham != None:
            self.H(Ham)
    
    def H(self,h):
        # construct vars for whole length of system
        full_ham_vars = [pstr(var.expr + ssum(Ilst(self.N-len(var.expr))),coeff = var.cf) for var in h]
        # compute hashes
        ham_hashes = [cyclic_hash(x.expr) for x in full_ham_vars]
        cfs = [x.cf for x in full_ham_vars]
        for i in range(len(full_ham_vars)):
            ind = self.dual_hashes.index(ham_hashes[i])
            self.c[ind] = cfs[i]
        return self.c
    
    def solve(self):
        x = cp.Variable(self.n_duals)
        
        Fm = self.Fmats
        
        M = Fm[self.idhash] + sum([Fm[self.dual_hashes[i]]*x[i] for i in range(self.n_duals)])
        constraints = [M>>0]

        t = time()
        c = self.c
        objective = cp.Minimize(c.T @ x)

        sdp = cp.Problem(objective,constraints)

        result = sdp.solve(solver = cp.MOSEK,verbose = False)
        optimal = x.value
        # rescale by size of orbit if needed
#         optimal = np.array([self.N/(len(self.orbits[i]))*x for i,x in enumerate(optimal)])

        self.result = (result,optimal)
        self.solved = True

        tt = time()
        print('solve time',tt-t)

        return result,optimal


    def slack_solve(self,energy):
        x = cp.Variable(self.n_duals)
        t = cp.Variable() # slack variable

        Fm = self.Fmats
        
        M = Fm[self.idhash] + sum([Fm[self.dual_hashes[i]]*x[i] for i in range(self.n_duals)])
        constraints = [M>>0]

        c = self.c
        constraints += [c.T @ x == energy]

        objective = cp.Maximize(t)

        sdp = cp.Problem(objective,constraints)

        result = sdp.solve(solver = cp.MOSEK)
        optimal = x.value
        # rescale by size of orbit if needed
#         optimal = np.array([self.N/(len(self.orbits[i]))*x for i,x in enumerate(optimal)])

        self.result = (result,optimal)
        self.solved = True

        return result,optimal


if __name__ == '__main__':
	N = 10
	xx = pstr('XX')
	z = pstr('Z')
	bas = ['X','Y','Z','XX','YY','ZZ']

	t1 = time()
	prob = pauliSDP(bas,N,Ham = [-1*xx,-1*z])
	res,opt = prob.solve()
	t2 = time()
	print("result:",res)
	print("total time:",t2-t1)



	
