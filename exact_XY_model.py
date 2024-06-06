# Exact solutions and exact diagonalization
import numpy as np

from scipy.sparse import kron, identity
from scipy.linalg import eigh
from scipy.integrate import quad,quadrature
from scipy.stats import linregress

################### 
# XY MODEL

# H = -1/2(1+g)XX - 1/2(1-g)YY - hZ

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
	res,err = quad(integrand,0,np.pi)
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

def XY_corrs(R,h,gam,N):
	R = np.array(R,dtype = int)
	res = {}
	res['XX'] = np.array([XY_exXX(r,h,gam,N) for r in R])
	res['YY'] = np.array([XY_exYY(r,h,gam,N) for r in R])
	res['ZZ'] = np.array([XY_exZZ(r,h,gam,N) for r in R])
	return res

def XY_getCorrs(name,R,h,gam,N):
	if name == 'XX':
		return np.array([XY_exXX(r,h,gam,N) for r in np.array(R)])
	if name == 'YY':
		return np.array([XY_exYY(r,h,gam,N) for r in np.array(R)])
	if name == 'ZZ':
		return np.array([XY_exZZ(r,h,gam,N) for r in np.array(R)])


	
################### 
# ISING MODEL

def exact_IsingHam(N,h,periodic = True):
	# Define Pauli matrices
	J =1
	sigma_z = np.array([[1, 0], [0, -1]])
	sigma_x = np.array([[0, 1], [1, 0]])
	dim = 2**N  # Dimension of the Hilbert space
	H = np.zeros((dim, dim))
	
	# Define the interaction terms
	for i in range(N):
		term = 1
		for j in range(N):
			if j == i+1 or (periodic and i == N-1 and j == N-1):
				# Kronecker products to build the interaction terms sigma_z[i] * sigma_z[j]
				
				for k in range(N):
					if k == i or k == j:
						term = np.kron(term, sigma_x)
					else:
						term = np.kron(term, np.eye(2))
				H -= J * term  # Add the term to the Hamiltonian
	
	# Define the transverse field terms
	for i in range(N):
		term = 1
		for k in range(N):
			if k == i:
				term = np.kron(term, sigma_z)
			else:
				term = np.kron(term, np.eye(2))
		H -= h * term  # Add the term to the Hamiltonian

	return H

def exact_XX_op(N, n):
	"""
	Build the operator X_0 X_n for an N-spin system.
	
	Parameters:
	N (int): Total number of spins in the system.
	n (int): Index of the second X operator (first is always at index 0).
	
	Returns:
	numpy.ndarray: The matrix representing the X_0 X_n operator.
	"""
	# Define the Pauli X and identity matrices
	X = np.array([[0, 1], [1, 0]])
	I = np.eye(2)
	
	# Start with X on the first spin
	operator = X # if n == 0 else I
	
	# Construct the operator using Kronecker products
	for i in range(1, N):
		if i == n:
			operator = np.kron(operator, X)
		else:
			operator = np.kron(operator, I)
	
	return operator

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

def Ising_Hamiltonian(N, mu,periodic = True): # for exact diagonalization
	H = np.zeros((int(2**N), int(2**N)))
	for i in range(N):
		H_term = 1.0
		for j in range(N):
			if j == i:
				H_term = np.kron(H_term, -mu * sigma_z)
			else:
				H_term = np.kron(H_term, np.eye(2))
		H += H_term
	for i in range(N):
		H_term = 1.0
		for j in range(N):
			if j == i:
				H_term = np.kron(H_term, -sigma_x)
			elif j == (i + 1) % N and periodic:
				H_term = np.kron(H_term, -sigma_x)
			elif j == (i+1) and not periodic:
				H_term = np.kron(H_term, -sigma_x)
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

if __name__ == '__main__':

	print(XY_GS(1,1,20)/20)

