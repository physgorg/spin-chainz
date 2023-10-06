# Exact solutions and exact diagonalization
import numpy as np

from scipy.sparse import kron, identity
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.stats import linregress

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

def Ising_FSS_critGS(N,order = 2):
	res = -4/np.pi
	if order == 0: return res 
	if order == 1:
		return res - np.pi/(6*N**2)
	if order == 2:
		return res - np.pi/(6*N**2) - 7*np.pi**3/(1440*N**4)


def extrapolate_energy(N, a, b):
    return a + b/N 
    
def largeNextrapolate(system_sizes,energies):
    energies = [energies[i]/system_sizes[i] for i in range(len(energies))]
    params, _ = curve_fit(extrapolate_energy, system_sizes, energies)
    extrapolated_energy = params[0]
    return extrapolated_energy