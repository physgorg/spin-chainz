# spin-chainz
sdp algorithms for ground state of spin chains
in development

################################

pauliSDP defines custom classes for Pauli strings and their algebra

Automatically sets up SDP problem and solves

Current features:
	- Can add constraints of the form <[H,O]> = 0
	- Automatically implements translation invariance
	- Can extract correlators and fit critical exponents to 2pt functions

Features in progress: 
	- Systematic determination of good bases
	- Determination of OPE coefficients from the critical CFT
	- Fitting the correlation length as a function of mu
	- Anchoring some basis elements at site 0
	- Infinite-volume version

