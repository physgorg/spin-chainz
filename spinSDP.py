import numpy as np
from time import time
import cvxpy as cp
import matplotlib.pyplot as plt

import openfermion as op
from openfermion.transforms import normal_ordered
from openfermion.transforms import jordan_wigner, bravyi_kitaev,reverse_jordan_wigner
from openfermion.ops import QubitOperator,FermionOperator
from openfermion.utils import hermitian_conjugated

from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sp
from scipy.stats import linregress

import exact_XY_model as XY

##########################################
# add a handy method to the QubitOperator, FermionOperator classes

def op_coeff(self):
    return list(self.terms.items())[0][1]

QubitOperator.op_coeff = op_coeff

FermionOperator.op_coeff = op_coeff

def c(i):
    return FermionOperator(f'{i}')
def cdag(i):
    return FermionOperator(f'{i}^')

def com(a,b):
    return a*b - b*a

##########################################
# Hash functions, for enforcing translational or cyclic invariance

def cyclic_hash(s): # cyclic invariant hash function
    '''
    Given a string s, outputs an integer hash value invariant under cyclic
    permutations of the string. Used to enforce cyclic translation invariance in the SDP.
    ''' 
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

def thermo_hash(s):
    '''
    Given a string s, outputs an integer hash value that is unique for each string
    and not invariant under cyclic permutations of the string. For open BC SDPs. 
    '''
    base = 7
    mod = 10**9 + 7
    hash_value = 0
    for i, char in enumerate(s):
        # ord(char) converts the character to its ASCII value
        # i is used to incorporate the position of the character in the string
        hash_value = (hash_value * base + ord(char) * (i + 1)) % mod
    return hash_value

def pauli_hash(pauli_op,bcs = None): # assume this is a product operator
    '''
    Generates an integral hash value for Pauli operator expectation values. 
    Enforces translation invariance throughout, and enforces cyclic invariance if N is not None.

    '''
    items = list(pauli_op.terms.items())
    if len(items)>1: 
        raise ValueError("Can only compute hash for a product operator.")
        return None
    term,coeff = items[0]

    if bcs is None:
        return term
    elif bcs[0] is None and bcs[1] == 'thermo': # in this case, enforce only translation invariance
        if len(term) == 0: # identity special case
            return thermo_hash('I')
        s = ''
        zero = term[0][0]
        for site,op in term:
            s += op+str(site-zero)
        return thermo_hash(s)

    elif bcs[1] == 'cyclic':
        N = bcs[0]
        s = ['I']*N
        if len(term) == 0: # identity special case
            return cyclic_hash(''.join(s))
        for site,op in term:
            s[site] = op
        return cyclic_hash(''.join(s))



def fermi_hash(fermi_op,N = None): # assume this is a product operator
    '''
    Generates an integral hash value for Fermi operator expectation values. 
    Does not enforce translation invariance at all. 

    '''
    fermi_op = normal_ordered(fermi_op)
    items = list(fermi_op.terms.items())
    if len(items)>1: 
        raise ValueError("Can only compute hash for a product operator.")
        return None
    term,coeff = items[0]
    return term

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

def XY_corrCompare(prob,mu,gam,title = None): # compare correlation functions to exact (finite-size) Ising solution
    corrs = prob.get_corrs()
    N = prob.N

    rv, XX = corrs['XX']
    rv, YY = corrs['YY']
    rv, ZZ = corrs['ZZ']
    ZZc = ZZ - corrs['Z']**2

    exact_corrs = XY.XY_corrs(rv,mu,gam,N)
    iXX = exact_corrs['XX']
    iYY = exact_corrs['YY']
    iZZ = exact_corrs['ZZ']

    fig,ax = plt.subplots(1,3,figsize = (15,5))
    x2,y2,z2 = ax

    x2.plot(rv,XX,label = 'SDP')
    x2.plot(rv,iXX,'k--',label = 'exact')
    x2.legend()
    x2.set_title(r"Correlator $\langle XX \rangle$; N = {}, $\mu = {},\ \gamma = {}$".format(N,mu,gam))
    x2.set_ylabel(r"value of corr. func")
    x2.set_xlabel(r"distance")

    y2.plot(rv,YY,label = 'SDP')
    y2.plot(rv,iYY,'k--',label = 'exact')
    y2.legend()
    y2.set_title(r"Correlator $\langle YY \rangle$; N = {}, $\mu = {},\ \gamma = {}$".format(N,mu,gam))
    # y2.set_ylabel(r"value of corr. func")
    y2.set_xlabel(r"distance")

    z2.plot(rv,ZZc,label = 'SDP')
    z2.plot(rv,iZZ,'k--',label = 'exact')
    z2.legend()
    z2.set_title(r"Correlator $\langle ZZ \rangle$; N = {}, $\mu = {},\ \gamma = {}$".format(N,mu,gam))
    # z2.set_ylabel(r"value of corr. func")
    z2.set_xlabel(r"distance")
    if title != None:
        fig.suptitle(title)
    plt.show()

#############################################
# main problem definition

class pauliSDP: # Pauli SDP problem class
    '''
    Constructs an SDP problem object. 
    Input: operator basis (list of pauliOps)
    Automatically constructs the LMI constraint.
    
    Methods: 
        - setup [creates a cvxpy problem and empty objective function]
        - H [adds a Hamiltonian to the problem and defines the energy as the objective]
        - generate_constraint [prepares a linear constraint <op> = rhs]
        - solve [solves the SDP, by default the energy problem]
    
    '''
    
    def __init__(self,operator_basis,
        h = None,
        bcs = None,
        v = False,
        timed = False,
        ):
    
        self.v = v # verbosity option
        self.timed = timed # printed time option
        if bcs is not None and not(bcs[0] is None and bcs[1] == 'thermo') and not (isinstance(bcs[0],int) and bcs[1] == 'cyclic'):
            raise TypeError("Not a valid boundary condition. Valid conditions are None, (None,'thermo'), (N,'cyclic').")
        self.bcs = bcs
    
        self.basis = np.array([QubitOperator('')]+operator_basis) # always put an identity operator in the basis
      
        self.Mshape = (len(self.basis),len(self.basis))
        if v:
            print("Matrix size:",self.Mshape)
    
        self.idhash = pauli_hash(self.basis[0],bcs = self.bcs)
    
        # Create the positivity matrix.
        # Since the objects have algebra all multiplication/commute/anticommute is done here
        t = time()
        self.Fmats = {}
        self.duals = []
        self.dual_hashes = []
        
        for i in range(len(self.basis)):
            for j in range(i, len(self.basis)):
                
                elem = hermitian_conjugated(self.basis[i]) * self.basis[j]  # main multiplication step here
                
                for operator in elem:  # elem = 2X(n)Z(m)
                    
                    hash_value = pauli_hash(operator, bcs=self.bcs)
                    coeff = operator.op_coeff()
                    
                    if hash_value not in self.Fmats.keys():  # if we have not got this variable yet
                        if hash_value != self.idhash:
                            self.duals.append(operator / coeff)
                            self.dual_hashes.append(hash_value)
                        self.Fmats[hash_value] = sp.lil_matrix((len(self.basis), len(self.basis)), dtype=complex)
                    self.Fmats[hash_value][i, j] = coeff
                    self.Fmats[hash_value][j, i] = np.conj(coeff)
        
        tt = time()
        if timed:
            print('find M, Fmats time =', tt - t)
        
        self.n_duals = len(self.duals)  # identity does not count as a dual
        
        if v:
            print("Number of dual variables:", self.n_duals)
        
        self.solved = False
        self.result = None
        self.opt_dict = None
        
        xvar = cp.Variable(self.n_duals)  # define vector of dual variables
        self.x = xvar
        
        Fm = self.Fmats  # matrices F_i which comprise M = F_0 + F_1*x1 + ...
        
        M = Fm[self.idhash].tocsc() + sum([Fm[self.dual_hashes[i]].tocsc() * xvar[i] for i in range(self.n_duals) if self.dual_hashes[i] in Fm.keys()])
        
        self.constraints = [M >> 0]
    
        c = cp.Parameter(self.n_duals,complex = True)

        self.c = c
    
        self.obj_func = cp.Minimize(cp.real(self.c.T @ self.x))
    
        sdp = cp.Problem(self.obj_func,self.constraints) # by setting up the problem now, we can easily change the obj func later
    
        self.problem = sdp

        self.obj_offset = 0
    
    def objective(self,objective_op):
        vector = np.zeros(self.n_duals,dtype = complex)
        for op in objective_op:
            # print('doing op',op)
            hash_val = pauli_hash(op,bcs = self.bcs)
            # print('op hash')
            if hash_val == self.idhash:
                self.obj_offset += op.op_coeff()
                continue
            try:
                ind = self.dual_hashes.index(hash_val)
                # print('corresponds to dual',self.duals[ind])
                vector[ind] += op.op_coeff()
            except ValueError:
                print("Objective not representable by dual variables.")
                print("Problem operator:",hash_val)
        self.c.value = vector

    def add_constraint(self,ops,rhs = 0):
        '''
        Adds a linear equality constraint to the SDP. 
        Inputs:
            - op [fermiOp] 
            - rhs [complex]
        Adds the constraint <op> = rhs as v @ c = b in cvxpy problem. 
        '''
        if isinstance(ops,QubitOperator):
            op = ops
            v = np.zeros(self.n_duals,dtype = complex)
            b = rhs
            
            for sub_op in op:
                hashval = pauli_hash(sub_op,bcs = self.bcs)
                coeff = sub_op.op_coeff()
                if hashval == self.idhash:
                    b -= coeff
                else:
                    try:
                        ind = self.dual_hashes.index(hashval)
                        v[ind] = coeff
                    except ValueError:
                        print()
                        print("Operator",sub_op, "wasn't found in basis.")
                        return None

            self.constraints += [v.T @ self.x == b]
            
            self.problem = cp.Problem(self.obj_func,self.constraints)
            
        elif isinstance(ops,list):
            to_add = []
            for op in ops:
                v = np.zeros(self.n_duals,dtype = complex)
                b = rhs

                op = normal_ordered(op)
                
                for sub_op in op:
                    hashval = fermi_hash(sub_op,bcs = self.bcs)
                    coeff = sub_op.op_coeff()
                    if hashval == self.idhash:
                        b -= coeff
                    else:
                        try:
                            ind = self.dual_hashes.index(hashval)
                            v[ind] = coeff
                        except ValueError:
                            print()
                            print("Operator",sub_op, "wasn't found in basis.")
                            continue

                to_add += [v.T @ self.x == b]

            self.constraints += to_add
            self.problem = cp.Problem(self.obj_func,self.constraints)
    
    def solve(self,**kwargs):
    
        t = time()
    
        if self.v: print("Solving SDP....")
    
        result = self.problem.solve(**kwargs)# /self.N
        optimal = self.x.value
    
        self.opt_dict = {self.dual_hashes[i]:optimal[i] for i in range(self.n_duals)}
    
        self.result = (result,optimal)
        if self.v: print("Result:",result)
        self.solved = True
    
        tt = time()
        if self.timed: print('solve time',tt-t)
    
        return result,optimal

    def get_opt_val(self,operator): # extract optimal value of operator from solved SDP
        if not self.solved:
            print("Problem is not solved.")
            return None
        value = 0
        for op in operator:
            hsh = pauli_hash(op,bcs = self.bcs)
            value += op.op_coeff()*self.opt_dict[hsh]
        return value

    def get_corrs(self):
        if self.solved == False:
            print("Problem is not solved.")
            return None

        corrdict = {}

        onepts = ['X','Y','Z']
        for operator in onepts:
            op = QubitOperator(operator+'0')
            hashval = pauli_hash(op,bcs = self.bcs)
            try:
                optval = self.opt_dict[hashval]
            except KeyError:
                print("Correlator",op,'not present in matrix.')
                continue
            if abs(optval) < 1e-10:
                optval = 0
            corrdict[operator] = optval

        twopts = ['XX','YY','ZZ']
        for operator in twopts:
            if (isinstance(self.bcs[0],int) and self.bcs[1] == 'cyclic'):
                N = self.bcs[0]
                sites = np.arange(N+1)
                optvals = np.zeros(N+1)
                optvals[0] = optvals[-1] = 1
                for i in range(1,N):
                    op = QubitOperator(f'{operator[0]}0 {operator[1]}{i}')
                    hashval = pauli_hash(op,bcs = self.bcs)
                    try:
                        optval = self.opt_dict[hashval]
                    except KeyError:
                        print("Correlator",op,'not present in matrix.')
                        continue
                    optvals[i] = optval
                data = [[int(x) for x in sites],optvals]
                corrdict[operator] = data

        return corrdict

class fermiSDP: # Pauli SDP problem class
    '''
    Constructs an SDP problem object. 
    Input: operator basis (list of fermiOps)
    Automatically constructs the LMI constraint.
    
    Methods: 
        - setup [creates a cvxpy problem and empty objective function]
        - objective [adds an objective function to the problem (usually a Hamiltonian)]
        - generate_constraint [prepares a linear constraint <op> = rhs]
        - solve [solves the SDP, by default the energy problem]
    
    '''
    
    def __init__(self,operator_basis,
        h = None,
        N = None,
        v = False,
        timed = False,
        ):
    
        self.v = v # verbosity option
        self.timed = timed # printed time option
        self.N = N
    
        self.basis = np.array([FermionOperator('')]+operator_basis) # always put an identity operator in the basis
      
        self.Mshape = (len(self.basis),len(self.basis))
    
        self.idhash = fermi_hash(self.basis[0],N = self.N)
    
        # Create the positivity matrix.
        # Since the objects have algebra all multiplication/commute/anticommute is done here
        t = time()
        self.duals = []
        self.dual_hashes = []
        self.Fmats = {}
        for i in range(len(self.basis)):
            for j in range(i,len(self.basis)):
    
                elem = normal_ordered(hermitian_conjugated(self.basis[i])*self.basis[j]) # main multiplication step here
                # print('elem',elem)
                for operator in elem: # elem = 2X(n)Z(m)
                    # print('original operator',operator)
                    hash_value = fermi_hash(operator,N = self.N)
                    coeff = operator.op_coeff()
                    
                    if hash_value not in self.Fmats.keys(): # if we have not got this variable yet
                        if hash_value != self.idhash:
                            # print('adding operator',operator/coeff,'with hash',hash_value)
                            self.duals.append(operator/coeff)
                            self.dual_hashes.append(hash_value)
                        self.Fmats[hash_value] = lil_matrix(self.Mshape, dtype=complex)
                    self.Fmats[hash_value][i,j] = coeff
                    self.Fmats[hash_value][j,i] = np.conj(coeff)
    
        tt = time()     
        if timed: print('find M, Fmats time =',tt - t)
    
        self.n_duals = len(self.duals) # identity does not count as a dual
    
        if v: print("Number of dual variables:",self.n_duals)
    
        self.solved = False
        self.result = None
        self.opt_dict = None


    
        xvar = cp.Variable(self.n_duals) # define vector of dual variables
        self.x = xvar
    
        Fm = self.Fmats # matrices F_i which comprise M = F_0 + F_1*x1 + ...

        F0 = self.Fmats[self.idhash].tocsr()
    
        M = Fm[self.idhash].tocsr() + sum([Fm[self.dual_hashes[i]].tocsr()*xvar[i] for i in range(self.n_duals) if self.dual_hashes[i] in Fm.keys()])
    
        self.constraints = [M>>0] # standard LMI semidefinite constraint
    
        c = cp.Parameter(self.n_duals)

        self.c = c
    
        self.obj_func = cp.Minimize(self.c.T @ self.x)
    
        sdp = cp.Problem(self.obj_func,self.constraints) # by setting up the problem now, we can easily change the obj func later
    
        self.problem = sdp

        self.obj_offset = 0
    
    def objective(self,objective_op):
        vector = np.zeros(self.n_duals)
        for op in objective_op:
            op = normal_ordered(op)
            # print('doing op',op)
            hash_val = fermi_hash(op,N = self.N)
            # print('op hash')
            if hash_val == self.idhash:
                self.obj_offset += op.op_coeff()
                continue
            try:
                ind = self.dual_hashes.index(hash_val)
                # print('corresponds to dual',self.duals[ind])
                vector[ind] += op.op_coeff()
            except ValueError:
                print("Objective not representable by dual variables.")
                print("Problem operator:",hash_val)
        self.c.value = vector

    def add_constraint(self,ops,rhs = 0):
        '''
        Adds a linear equality constraint to the SDP. 
        Inputs:
            - op [fermiOp] 
            - rhs [complex]
        Adds the constraint <op> = rhs as v @ c = b in cvxpy problem. 
        '''
        if isinstance(ops,FermionOperator):
            op = ops
            v = np.zeros(self.n_duals,dtype = complex)
            b = rhs

            op = normal_ordered(op)
            
            for sub_op in op:
                hashval = fermi_hash(sub_op,N = self.N)
                coeff = sub_op.op_coeff()
                if hashval == self.idhash:
                    b -= coeff
                else:
                    try:
                        ind = self.dual_hashes.index(hashval)
                        v[ind] = coeff
                    except ValueError:
                        print()
                        print("Operator",sub_op, "wasn't found in basis.")
                        return None

            self.constraints += [v.T @ self.x == b]
            
            self.problem = cp.Problem(self.obj_func,self.constraints)

        elif isinstance(ops,list):
            to_add = []
            missed = 0
            for op in ops:
                v = np.zeros(self.n_duals,dtype = complex)
                b = rhs

                op = normal_ordered(op)

                representable = True
                
                for sub_op in op:
                    if representable:
                        hashval = fermi_hash(sub_op,N = self.N)
                        coeff = sub_op.op_coeff()
                        if hashval == self.idhash:
                            b -= coeff
                        else:
                            try:
                                ind = self.dual_hashes.index(hashval)
                                v[ind] = coeff
                            except ValueError:
                                representable = False
                if not representable:
                    missed += 1
                else:
                    to_add += [v.T @ self.x == b]

            if missed > 0: print(f'{missed} constraints not representable.')
            self.constraints += to_add
            self.problem = cp.Problem(self.obj_func,self.constraints)
    
    def solve(self,include_Hcoms = False,**kwargs):
    
        t = time()
    
        if self.v: print("Solving SDP....")
    
        result = self.problem.solve(solver = cp.MOSEK,**kwargs)

        result += self.obj_offset
        
        optimal = self.x.value
    
        self.opt_dict = {self.dual_hashes[i]:optimal[i] for i in range(self.n_duals)}
    
        self.result = (result,optimal)
        if self.v: print("Result:",result)
        self.solved = True
    
        tt = time()
        if self.timed: print('solve time',tt-t)
    
        return result,optimal

    def get_opt_val(self,operator): # extract optimal value of operator from solved SDP
        if not self.solved:
            print("Problem is not solved.")
            return None
        value = 0
        operator = normal_ordered(operator)
        for op in operator:
            hsh = fermi_hash(op,N = self.N)
            cf = op.op_coeff()
            # print('op',op,'hash',hsh)
            try:
                value += cf*self.opt_dict[hsh]
            except KeyError:
                print("The operator",op,"is not one of the optimization variables.")
                return None
        return value
    

    def get_corrs(self):
        if self.solved == False:
            print("Problem is not solved.")
            return None

        corrdict = {}

        corrdict['Z'] = 1-2*self.get_opt_val(FermionOperator('0^ 0'))

        def sdpG(rr):
            # assume rr!=0 here
            if rr == 0:
                op = 2*cdag(0)*c(0) 
                return self.get_opt_val(op) -1
            elif rr > 0:
                op = cdag(0)*cdag(rr) - c(0)*cdag(rr) + cdag(0)*c(rr) - c(0)*c(rr)
                return self.get_opt_val(op)
            elif rr < 0:
                rr = abs(rr)
                op = cdag(rr)*cdag(0) - c(rr)*cdag(0) + cdag(rr)*c(0) - c(rr)*c(0)
                return self.get_opt_val(op)

        sites = np.arange(self.N +1)

        # do ZZ correlator
        zz_vals = [1 - corrdict['Z']**2] + [-1*sdpG(r)*sdpG(-r) for r in range(1,self.N)] + [1-corrdict['Z']**2]
        corrdict['ZZ'] = [sites,np.array(zz_vals)+corrdict['Z']**2] # construct connected correlator

        # do XX correlator
        xx_vals = [1] 
        for R in range(1,self.N):
            row = lambda n: np.array([sdpG(rr) for rr in (n + 2 - np.arange(1,R+1))])
            mat = np.vstack([row(k) for k in range(R)])
            xx_vals.append(np.linalg.det(mat))
        xx_vals += [1]
        corrdict['XX'] = [sites,xx_vals]

        # do YY correlator
        yy_vals = [1] 
        for R in range(1,self.N):
            row = lambda n: np.array([sdpG(rr) for rr in (n - np.arange(1,R+1))])
            mat = np.vstack([row(k) for k in range(R)])
            yy_vals.append(np.linalg.det(mat))
        yy_vals += [1]
        corrdict['YY'] = [sites,yy_vals]
        

        return corrdict

        


###############################################
# Some useful functions for building problems

def pauli_onept_basis(N):
    letters = ['X','Y','Z']
    sites = range(N)
    basis = [QubitOperator((site,letter)) for site in sites for letter in letters]
    return basis

def pauli_twopt_basis(N,anchored = False):
    basis = pauli_onept_basis(N)
    newops = ['XX','YY','ZZ','XY','YZ','XZ','ZX','ZY','YX']
    if anchored:
        for corr in newops:
            basis.append(QubitOperator(f'{corr[0]}0 {corr[1]}1'))
    elif not anchored:
        for corr in newops:
            for i in range(N):
                basis.append(QubitOperator(f'{corr[0]}{i} {corr[1]}{(i+1)%N}'))
    return basis

def pauli_TFIM_Ham(N,gamma = 1,mu = 1):
    return np.sum([-1/2*(1+gamma)*QubitOperator(f'X{i} X{(i+1) % N}') 
        -1/2*(1-gamma)*QubitOperator(f'Y{i} Y{(i+1) % N}') 
        - mu*QubitOperator(f'Z{i}') for i in range(N)])

def fermi_onept_basis(N):
    basis = []
    for i in range(N):
        basis.append(c(i))
        basis.append(cdag(i))
    return basis

def fermi_TFIM_ham(N,h,gamma=1,p = 0): # p controls fermion parity (PBC vs ABC for fermions)
    hobc = -1*np.sum([cdag(j)*c(j+1) + gamma*cdag(j)*cdag(j+1) + cdag(j+1)*c(j) + gamma*c(j+1)*c(j) for j in range(N-1)])
    hobc += h*np.sum([2*cdag(j)*c(j) - 1 for j in range(N)])
    hpbc = hobc + (-1)**p*(cdag(N-1)*c(0) + gamma*cdag(N-1)*cdag(0) + cdag(0)*c(N-1) + gamma*c(0)*c(N-1))
    return hpbc


if __name__ == '__main__':

    N = 20

    operator_basis = twopt_basis(N,anchored = True)

    prob = pauliSDP(operator_basis,N = N,timed = True,v = True)

    ising_ham_density = TFIM_Ham(N,1)
    prob.objective(ising_ham_density)

    res,opt = prob.solve()

    exact = XY.XY_GS(1,1,N)/N

    print("Exact:",exact)
    print("SDP:",res)
    print("Percent error:",abs((exact-res)/exact)*100)

    XY_corrCompare(prob,1,1)










