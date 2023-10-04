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

def frob(A,B): # sympy Frobenius inner product
    return sp.trace(Dagger(A)*B)

def roll_list(input_list, n): # numpy roll method but for list
    length = len(input_list)
    n = n % length  
    return input_list[-n:] + input_list[:-n]
        
def proll(x,n): # roll a pstring object
    s = roll_list(x.ops,n)
    s = ssum(s)
    return pstring(s)
 
def p(x): # shorthand func for pstring
    return pstring(x)
    
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
    # t0 = time()
    atoms = M.atoms(pstring)
    rows,cols = M.shape
    cf = sp.zeros(rows,cols)
    if var not in atoms:
        return 0
    else:
        for i in range(rows):
            for j in range(cols):
                element = M[i, j]
                if element.has(var):
                    cf[i,j] = element.coeff(var)
    # t1 = time()
    # print("Mcoeff time",t1-t0)
    return cf


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
        else:
            return super().__mul__(other) # otherwise inherit sympy mult
        
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

        self.basis = constructBasis(basis_ops,N)
        
        if v: print("Operator basis:"); LatDisp(self.basis.T)
        
        # Since the objects have algebra all multiplication/commute/anticommute is done here
        self.M = sp.Matrix(self.basis * self.basis.T)
        
        if v: print("Correlation matrix:"); LatDisp(self.M)
        
        self.all_vars = list(self.M.atoms(pstring))
        self.all_vars.remove(pID(self.N))
        
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
           
                

        self.n_duals = len(self.orbits)
        if v: print("Number of dual variables:",self.n_duals)

        self.solved = False
        self.res = None
        
        
        
        # construct PSD in the dual perspective
        rows,cols = self.M.shape
        self.Fmats = [np.zeros(self.M.shape,dtype = complex)]
        for orbit in self.orbits:
            cf = sp.zeros(rows,cols)
            for x in orbit:
                cf += getMcoeff(self.M,x)
            self.Fmats.append(np.array(cf,dtype = complex))

        v = self.M.xreplace({a:0 for a in self.all_vars})
        v = v.xreplace({pID(N):1})
        self.Fmats[0] = np.array(v)
        
        self.c = np.zeros((self.n_duals)) # objective function-to-be

        if Ham != None:
            self.H(Ham)
    
    def H(self,h):
        ham_vars = h.atoms(pstring)
        cfdict = h.as_coefficients_dict()
        cfs = [cfdict[var] for var in ham_vars]
        
        full_ham_vars = [pstring(var.expr + ssum(Ilst(self.N-len(var.expr)))) for var in ham_vars]
        
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
    
    def solve(self):
        x = cp.Variable(self.n_duals)

        Fm = self.Fmats
        M = Fm[0] + sum([Fm[i+1]*x[i] for i in range(self.n_duals)])
        constraints = [M>>0]

        c = self.c
        objective = cp.Minimize(c.T @ x)

        sdp = cp.Problem(objective,constraints)

        result = sdp.solve(solver = cp.MOSEK,verbose = False)
        optimal = x.value

        self.result = (result,optimal)
        self.solved = True

        return result*self.N,optimal


if __name__ == '__main__':
    zz = pstring('ZZ')
    x = pstring('X')

    mu = 1

    h = -1*zz - mu*x

    N = 8

    t = time()
    yz = pSDP(['I','X','ZZ'],N,Ham = h)
    print(yz.basis)
    yz_res,_ = yz.solve()
    tt = time()

    print("yz time",tt-t)
    print('yz res',yz_res)

#########################################################################

# AUXILIARY STUFF

################### 
# ISING MODEL

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]]) 

def Ising_Hamiltonian(N, mu): # for exact diagonalization
    H = np.zeros((int(2**N), int(2**N)))
    for i in range(N):
        H_term = 1.0
        for j in range(N):
            if j == i:
                H_term = np.kron(H_term, -mu * sigma_x)
            else:
                H_term = np.kron(H_term, np.eye(2))
        H += H_term
    for i in range(N):
        H_term = 1.0
        for j in range(N):
            if j == i:
                H_term = np.kron(H_term, -sigma_z)
            elif j == (i + 1) % N:
                H_term = np.kron(H_term, -sigma_z)
            else:
                H_term = np.kron(H_term, np.eye(2))
        H += H_term
    return H

def Ising_exactD(N,mu): # exact diagonalization
    H = Ising_Hamiltonian(N,mu)
    eigenvalues, eigenvectors = eigh(H)
    return eigenvalues[0]

def Ising_analytical_solution(mu, k): # exact dispersion relation
    return np.sqrt(1 + mu**2 - 2 * mu * np.cos(k))

def Ising_analytical_GS(mu): # get exact ground state
    integral, _ = quad(lambda k: Ising_analytical_solution(mu, k), 0, np.pi)
    analytical_energy = -integral / np.pi
    return analytical_energy

def extrapolate_energy(N, a, b):
    return a + b/N 
    
def largeNextrapolate(system_sizes,energies):
    energies = [energies[i]/system_sizes[i] for i in range(len(energies))]
    params, _ = curve_fit(extrapolate_energy, system_sizes, energies)
    extrapolated_energy = params[0]
    return extrapolated_energy

def cleanUp(arr,tol = 1e-15): # make array presentable
    dec = int(-1*np.log10(tol))
    real = np.real(arr)
    imag = np.imag(arr)
    real[np.abs(real) < tol] = 0
    imag[np.abs(imag) < tol] = 0
    real = np.round(real,decimals = dec)
    imag = np.round(imag,decimals = dec)
    return real + 1j*imag
    

            



