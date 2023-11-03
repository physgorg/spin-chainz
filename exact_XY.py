# Exact solutions and exact diagonalization
import numpy as np

from scipy.sparse import kron, identity
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.stats import linregress

################### 
# XY MODEL

def omega(k,h,g):
    if h == 1 and g == 1:
        return np.sqrt(2-2*np.cos(k))
    else:
        return np.sqrt((h - np.cos(k))**2 + g**2*np.sin(k)**2)
    
def kmodes(N):
    ns = np.arange(1,N/2 + 1)
    ks = (2*ns - 1)*np.pi/N
    return ks

# Finite size version
def XY_GS(h,gam,N):
    k = kmodes(N)
    return -2*np.sum(omega(k,h,gam))

def XY_GR(R,h,g,N):
    ks = kmodes(N)
    summand = np.cos(ks*R)*(h - np.cos(ks))/omega(ks,h,g) - g*np.sin(ks*R)*np.sin(ks)/omega(ks,h,g)
    summed = np.sum(summand)
    return -2/N*summed

def XY_exZ(h,gam,N):
    return -1*XY_GR(0,h,gam,N)

def XY_exZZ(R,h,gam,N): # keep in mind this is the connected version
    if R % N == 0 : return 1-XY_exZ(h,gam,N)**2
    return -1*XY_GR(R,h,gam,N)*XY_GR(-1*R,h,gam,N)

def XY_exXX(R,h,gam,N):
    if R % N == 0 : return 1
    # row = lambda n: np.array([XY_GR(rr,h,gam,N) for rr in (n - np.arange(1,R+1))])
    row = lambda n: np.array([XY_GR(rr,h,gam,N) for rr in (n + 2 - np.arange(1,R+1))])
    mat = np.vstack([row(k) for k in range(R)])
    return np.linalg.det(mat)

def XY_exYY(R,h,gam,N):
    if R % N == 0 : return 1
    # row = lambda n: np.array([XY_GR(rr,h,gam,N) for rr in (n + 2 - np.arange(1,R+1))])
    row = lambda n: np.array([XY_GR(rr,h,gam,N) for rr in (n - np.arange(1,R+1))])
    mat = np.vstack([row(k) for k in range(R)])
    return np.linalg.det(mat)

# in thermodynamic limit
def XY_GS_thermo(h,gam):
    integrand = lambda x: omega(x,h,gam)
    res,err = quad(integrand,0,2*np.pi)
    return -1/(2*np.pi)*res

# MAGNETIZATION
def XY_exZ_thermo(h,gam):
    integrand = lambda x: (h - np.cos(x))/omega(x,h,gam)
    res,err = quad(integrand,0,np.pi)
    return 1/(np.pi)*res
    
# TWO POINT CORRELATORS
def XY_GR_thermo(R,h,g):
    integrand = lambda x: np.cos(x*R)*(h-np.cos(x))/omega(x,h,g) - g*np.sin(x*R)*np.sin(x)/omega(x,h,g)
    res,err = quadrature(integrand,0,np.pi)
    return -1/np.pi*res


def XY_exZZ_thermo(R,h,gam): # keep in mind this is the connected version
    return -1*XY_GR_thermo(R,h,gam)*XY_GR_thermo(-1*R,h,gam)


def XY_exXX_thermo(R,h,gam):
    row = lambda n: np.array([XY_GR_thermo(rr,h,gam) for rr in (n + 2 - np.arange(1,R+1))])
    # row = lambda n: np.array([XY_GR_thermo(rr,h,gam) for rr in (n - np.arange(1,R+1))]) OLD
    mat = np.vstack([row(k) for k in range(R)])
    return np.linalg.det(mat)


def XY_exYY_thermo(R,h,gam):
    row = lambda n: np.array([XY_GR_thermo(rr,h,gam) for rr in (n - np.arange(1,R+1))])
    # row = lambda n: np.array([XY_GR_thermo(rr,h,gam) for rr in (n + 2 - np.arange(1,R+1))]) OLD
    mat = np.vstack([row(k) for k in range(R)])
    return np.linalg.det(mat)

def getCritExp(x,y,N = None):
    if N == None:
        reg = linregress(np.log(np.array(x)),np.log(np.array(y)))
        return reg.slope*-1/2
    else:
        reg = linregress(np.log(np.sin(np.pi*np.array(x)/N)**2),np.log(np.array(y)))
        return reg.slope*-1
    
################### 
# ISING MODEL

def Ising_corrs(h,N): # get exact correlation functions for all sites
    Rvals = np.arange(N+1) # distances from site
    res = {}
    res['X'] = [0]
    res['Y'] = [0]
    res['Z'] = [XY_exZ(h,1,N)]
    res['XX'] = (Rvals,np.array([XY_exXX(r,h,1,N) for r in Rvals]))
    res['YY'] = (Rvals,np.array([XY_exYY(r,h,1,N) for r in Rvals]))
    res['ZZ'] = (Rvals,np.array([XY_exZZ(r,h,1,N) for r in Rvals]))
    return res

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

def Ising_FSS_critGS(N,order = 2):
    res = -4/np.pi
    if order == 0: return res 
    if order == 1:
        return res - np.pi/(6*N**2)
    if order == 2:
        return res - np.pi/(6*N**2) - 7*np.pi**3/(1440*N**4)


# def extrapolate_energy(N, a, b):
#     return a + b/N 
    
# def largeNextrapolate(system_sizes,energies):
#     energies = [energies[i]/system_sizes[i] for i in range(len(energies))]
#     params, _ = curve_fit(extrapolate_energy, system_sizes, energies)
#     extrapolated_energy = params[0]
#     return extrapolated_energy
if __name__ == '__main__':

    XY_exXX(5,1,1,8)

