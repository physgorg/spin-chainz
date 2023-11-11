# old

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import cvxpy as cp

from sympy.physics.paulialgebra import Pauli
from sympy.physics.quantum.dagger import Dagger
from IPython.display import display, Latex

from scipy.sparse import kron, identity
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import linregress

from time import time

######################################################################

# FUNCTIONS

###### pauli strings ######
def pauli(j): # pauli matrix by vec index
    if j == 0:
        return 1
    elif j < 4:
        return Pauli(j)
    
def toLetter(pmat): # convert pauli matrix object to string letter
    ld = ['I','X','Y','Z']
    for i in range(4):
        if pmat - pauli(i) == 0:
            return ld[i]

def ssum(lst): # "sum" of list of strings
    return ''.join(lst)

def Ilst(n): 
    return ['I' for i in range(n)]

def pID(n): # pauli string identity
    ls = Ilst(n)
    ls = ssum(ls)
    return pstring(ls)

def p(x): # shorthand func for pstring
    return pstring(x)

###### rolling & ZN orbit stuff ######

def roll_list(input_list, n): # numpy roll method but for list
    length = len(input_list)
    n = n % length  
    return input_list[-n:] + input_list[:-n]
        
def proll(x,n): # roll a pstring object
    s = roll_list(x.ops,n)
    s = ssum(s)
    return pstring(s)


def F_revolve(arr,N,L,withIdentity):
    idL = np.identity(L)
    zeroL = np.zeros((L,L))
    sh = (N*L,N*L)
    
    if withIdentity:
        zer = np.array([[arr[0,0]]]) # identity corner
        v = arr[0,1:] # identity times ops
        subF = arr[1:,1:]
        revved = F_revolve(subF,N,L,False) # orbit the LN x LN portion
        
        # orbit the identity parts
        vorber = np.block([[np.eye(L) for j in range(N)] for i in range(N)])
        v = np.dot(vorber,v)
        v = np.array([v])
        
        vc = v.conj()
        res = np.block([[zer,v],[vc.T,revved]]) # re-blockify
        return res
        
    else:

        res = np.zeros(sh,dtype = complex)
        P = np.block([[1 if j == (i + 1) % N else 0 for j in range(N)] for i in range(N)])
        res += arr
        for n in range(1,N):
            pn = np.linalg.matrix_power(P,n)
            pn = np.kron(pn,idL)
            term = np.dot(pn.T,np.dot(arr,pn))
            res += term
        return res

###### sdp basis stuff ######

def frob(A,B): # sympy Frobenius inner product
    return sp.trace(Dagger(A)*B)
    
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
    operators = [pstring(x) for x in operators]
    if withIdentity: 
        return sp.Matrix([pstring(ssum(Ilst(N)))] + operators)
    else:
        return sp.Matrix(operators) 

def getMcoeff(M,var): # get pstring matrix coefficient
    atoms = M.atoms(pstring)
    rows,cols = M.shape
    cf = np.zeros((rows,cols),dtype = complex)
    for i in range(rows):
        for j in range(i,cols):
            element = M[i, j]
            if element.has(var):
                cf[i,j] = complex(element.coeff(var))
    cf = (cf + cf.conj().T)

    return cf

def pcom(a,b):
    # the following lines put in bilinearity by hand
    if isinstance(a,sp.Add):
        return sum([pcom(x,b) for x in a.args])
    elif isinstance(b,sp.Add):
        return sum([pcom(a,x) for x in b.args])
    elif isinstance(a,sp.Mul):
        cf = sp.prod(a.args[:-1])
        return pcom(a.args[-1],b)*cf
    elif isinstance(b,sp.Mul):
        cf = sp.prod(b.args[:-1])
        return pcom(a,b.args[-1])*cf
    elif isinstance(a,pstring) and isinstance(b,pstring):
        if len(a.expr) != len(b.expr):
            raise ValueError("Can't commutate Pauli strings of different lengths")
        else:
            return a*b - b*a
    


def LatDisp(matrix): # Convert each row of the matrix to its LaTeX representation
    matrix = sp.Matrix(matrix)
    if matrix.shape[-1] == 1:
        matrix = sp.Matrix([matrix])
    rows_latex = []
    for row in matrix.tolist():
        rowstrs = []
        for element in row:
            if isinstance(element,pstring):
                rowstrs.append(element.lat())
            elif isinstance(element,sp.core.mul.Mul):
                cf = sp.prod(element.args[:-1])
                ps = element.args[-1].lat()
                if cf.is_real:
                    if cf == 1 and len(ps) > 0:
                        cf = ''
                    else:
                        cf = format(float(cf),'.10g')
                        
                    toapp = str(cf)+ps
                    rowstrs.append(toapp)
                else:
                    cf = format(float(-1j*cf),'.10g')
                    rowstrs.append(str(cf)+'i'+ps)
        row_latex = " & ".join(rowstrs)
        rows_latex.append(row_latex)
    
    # Join the rows with '\\' and enclose in a bmatrix environment
    latex_matrix = r"\begin{bmatrix} " + r" \\ ".join(rows_latex) + r" \end{bmatrix}"
    
    # Display the LaTeX matrix
    display(Latex(latex_matrix))


##########################################################################

# CLASSES

class pstring(sp.Symbol): # Pauli string class
    
    def __new__(cls, name, **assumptions):
        # Ensure the symbol is noncommutative
        assumptions['commutative'] = False
        return sp.Symbol.__new__(cls, name, **assumptions)
    
    def __init__(self,expr):
        super().__init__()  # Call the parent's __init__
        
        self.ops = list(expr) # list of strings repn
        
        self.odict = {'I':pauli(0),'X':pauli(1),'Y':pauli(2),'Z':pauli(3)} # ops corresponding to symbols
        oplist = [self.odict[x] for x in self.ops]
        
        self.v = sp.Matrix(oplist).T # vector of Pauli objects
        self.expr = expr # string expression
        
        self.n = len(self.ops)
        
        self.ids = self.ops.count("I")
        self.xs = self.ops.count("X")
        self.ys = self.ops.count("Y")
        self.zs = self.ops.count("Z")

        self.counts = np.array([self.ids,self.xs,self.ys,self.zs])
        
    def __mul__(self, other): # multiplication of ops happens site-wise
        if isinstance(other, pstring): # if we are doing pauli string mat-mul,
            prodtns = self.v.multiply_elementwise(other.v)
            const = 1
            res = ''
            for op in prodtns:
                cf = np.prod([x if not isinstance(x,Pauli) else 1 for x in op.args])
                const = const * cf
                res += toLetter(op/cf)


            return const*pstring(res)
        
        elif isinstance(other,sp.Mul): # if multiplying by an expression
            cf = sp.prod(other.args[:-1])
            pstr = other.args[-1]
            prod = pstr*self
            return prod*cf

        elif isinstance(other,sp.Add): # if multiplying an expression, distribute
            return sum([super(pstring, self).__mul__(o) for o in other.args])
        else:
            return super().__mul__(other) # otherwise inherit sympy multx
        
    def tp(self,other): # shorthand tensor product
        if isinstance(other,pstring):
            return pstring(self.expr+other.expr)
        
    def lat(self): # render as latex-ready string
        elems = []
        if self.expr == ssum(Ilst(self.n)): # if pure identity,
            return 'I'
        else:
            for i,o in enumerate(self.ops):
                if o != 'I':
                    elems.append(o+'_{}'.format(i+1))
            dstr = ssum(elems)
            return dstr
    
    def disp(self): # display nicely
        display(Latex('$'+lat(self)+'$'))
 
        
    



class pSDP: # spin chain SDP class
    
    def __init__(self,basis_ops,N,Ham = None,v = False):
        
        self.N = N

        t = time()
        self.basis = constructBasis(basis_ops,N)
        tt = time()
        print('constructBasis time =',tt-t)
        
        self.withIdentity = ('I' in basis_ops)
        if self.withIdentity:
            self.L = len(basis_ops) - 1
        else:
            self.L = len(basis_ops)
        
        if v: print("Operator basis:"); LatDisp(self.basis.T)
        
        # Since the objects have algebra all multiplication/commute/anticommute is done here
        t = time()
        self.M = sp.Matrix(self.basis * self.basis.T)
        tt = time()
        print('find M time =',tt -t)
        
        if v: print("Correlation matrix:"); LatDisp(self.M)
        
        # get initial set of dual variables
        self.all_vars = list(self.M.atoms(pstring))
        self.all_vars.remove(pID(self.N))
        
        t = time()
        # Group by ZN orbits
        self.orbits = []
        duals_set = set(self.all_vars)
        while duals_set:
            x = duals_set.pop()  # Remove and return an arbitrary element from duals_set
            orbit = [x]
            for n in range(1, N+1):
                y = proll(x, n)
                if y in duals_set:
                    orbit.append(y)
                    duals_set.remove(y)  # Remove y from duals_set as it's already in orbit
            
            self.orbits.append(orbit)
        tt = time()
        print("orbits time =",tt-t)

        self.n_duals = len(self.orbits)
        if v: print("Number of dual variables:",self.n_duals)

        self.reps = [x[0] for x in self.orbits] # class reps of each orbit

        self.Hcom_ops = []

        self.solved = False

        self.result = None
        
        # construct PSD in the dual perspective
        t = time()
        self.Fmats = [np.zeros(self.M.shape,dtype = complex)]
        for representative in self.reps: # for each class representative
            cf = np.zeros(self.M.shape,dtype = complex)
            Q = getMcoeff(self.M,representative) # get single coeff 
            revolved_Q = F_revolve(Q,self.N,self.L,self.withIdentity)
            self.Fmats.append(revolved_Q)

        v = self.M.xreplace({a:0 for a in self.all_vars})
        v = v.xreplace({pID(N):1})
        self.Fmats[0] = np.array(v)

        tt = time()
        print("build Fmats time =",tt-t)
        
        self.c = np.zeros((self.n_duals)) # objective function-to-be

        if Ham != None:
            self.H(Ham)
    
    def H(self,h):
        ham_vars = h.atoms(pstring)
        cfdict = h.as_coefficients_dict()
        cfs = [cfdict[var] for var in ham_vars]
        
        full_ham_vars = [pstring(var.expr + ssum(Ilst(self.N-len(var.expr)))) for var in ham_vars]
        
        self.fullH = 0
        for n in range(self.N):
            single_site_h = sum([proll(full_ham_vars[i],n)*cfs[i] for i in range(len(full_ham_vars))])
            self.fullH += single_site_h

        all_ham_cfs = []
        
        for orbit in self.orbits:
            for k,var in enumerate(full_ham_vars):
                if var in orbit:
                    cf = cfs[k]
                    break
                else:
                    cf = 0
            
            all_ham_cfs.append(cf)
            
        self.c = np.array(all_ham_cfs)
        return self.c

    def add_Hcom_cnstr(self,op):
        self.Hcom_ops.append(op)

    def Hcom(self,newop):
        if self.fullH == None:
            print("ERROR: No Hamiltonian specified.")
            return 0
        current_ham = self.fullH
        L = self.N
        newop = pstring(newop + ssum(Ilst(L - len(newop))))
        return sp.expand(-1j*pcom(current_ham,newop)) # factor of i for Hermiticity. Guarantees real coeffs for operators (?) NOT SURE IF ALWAYS TRUE
    
    def slack_solve(self,energy):
        x = cp.Variable(self.n_duals)
        t = cp.Variable() # slack variable

        Fm = self.Fmats
        M = Fm[0] + sum([Fm[i+1]*x[i] for i in range(self.n_duals)])
        Id = np.identity(M.shape[0])
        constraints = [(M -t*Id)>>0]

        for op in self.Hcom_ops: # add commutator constraints directly, if they are included
            res = self.Hcom(op)
            variables = list(res.atoms(pstring))
            coeffs = res.as_coefficients_dict()
            coeffs = [coeffs[var] for var in variables]
            orbit_vars = []
            for var in variables:
                found = False
                for i,orb in enumerate(self.orbits):
                    if var in orb:
                        orbit_vars.append(x[i])
                        found = True
                        continue
            if found:
                constraints += [sum([coeffs[i]*orbit_vars[i] for i in range(len(coeffs))]) == 0]
            elif not found:
                print("Not all operators in commutator constraint are contained in M")

        # set energy to specific value
        c = self.c
        constraints += [c.T @ x == energy]

        objective = cp.Maximize(t)

        sdp = cp.Problem(objective,constraints)

        result = sdp.solve(solver = cp.MOSEK)
        optimal = x.value
        # rescale by size of orbit if needed
        optimal = np.array([self.N/(len(self.orbits[i]))*x for i,x in enumerate(optimal)])

        self.result = (result,optimal)
        self.solved = True

        return result,optimal


    def solve(self):
        x = cp.Variable(self.n_duals)
        
        Fm = self.Fmats
        M = Fm[0] + sum([Fm[i+1]*x[i] for i in range(self.n_duals)])
        constraints = [M>>0]

        for op in self.Hcom_ops: # add commutator constraints directly, if they are included
            res = self.Hcom(op)
            variables = list(res.atoms(pstring))
            coeffs = res.as_coefficients_dict()
            coeffs = [coeffs[var] for var in variables]
            orbit_vars = []
            for var in variables:
                found = False
                for i,orb in enumerate(self.orbits):
                    if var in orb:
                        orbit_vars.append(x[i])
                        found = True
                        continue
            if found:
                constraints += [sum([coeffs[i]*orbit_vars[i] for i in range(len(coeffs))]) == 0]
            elif not found:
                print("Not all operators in commutator constraint are contained in M")

        t = time()
        c = self.c
        objective = cp.Minimize(c.T @ x)

        sdp = cp.Problem(objective,constraints)

        result = sdp.solve(solver = cp.MOSEK,verbose = False)
        optimal = x.value
        # rescale by size of orbit if needed
        optimal = np.array([self.N/(len(self.orbits[i]))*x for i,x in enumerate(optimal)])

        self.result = (result,optimal)
        self.solved = True

        tt = time()
        print('solve time',tt-t)

        return result*self.N,optimal


if __name__ == '__main__':


    zz = pstring('ZZ')
    x = pstring('X')

    mu = 1

    h = -1*zz - mu*x

    N = 8

    t = time()
    yz = pSDP(['I','X','Y','Z'],N,Ham = h)


    # print(yz.Hcom('X'))
    # print(yz.basis)
    yz_res,_ = yz.slack_solve(1.2)
    tt = time()

    print("yz time",tt-t)
    print('yz res',yz_res)

#########################################################################

# AUXILIARY STUFF



def cleanUp(arr,tol = 1e-15): # make array presentable
    dec = int(-1*np.log10(tol))
    real = np.real(arr)
    imag = np.imag(arr)
    real[np.abs(real) < tol] = 0
    imag[np.abs(imag) < tol] = 0
    real = np.round(real,decimals = dec)
    imag = np.round(imag,decimals = dec)
    return real + 1j*imag
    

            