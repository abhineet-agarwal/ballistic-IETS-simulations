import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# Constants (all MKS, except energy which is in eV)
hbar = 1.06e-34  # Reduced Planck's constant (J·s)
q = 1.6e-19  # Elementary charge (C)
epsil = 10 * 8.85E-12  # Dielectric constant (F/m)
kT = 0.025  # Thermal energy at room temperature (eV)
m = 0.25 * 9.1e-31  # Effective mass of electron (kg)
n0 = 2 * m * kT * q / (2 * np.pi * hbar**2)  # Electron density parameter

# Inputs
a = 3e-10  # Lattice constant (m)
t = hbar**2 / (2 * m * a**2 * q)  # Hopping parameter (eV)
beta = q * a**2 / epsil  # Scaled inverse dielectric constant
Np = 100  # Total number of sites
XX = a * 1e9 * np.arange(1, Np + 1)  # Spatial grid (nm)

# Hamiltonian matrix
T = 2 * t * np.diag(np.ones(Np)) - t * np.diag(np.ones(Np - 1), 1) - t * np.diag(np.ones(Np - 1), -1)
N1 = 2
N2 = 98
UB1= 2*t
UB2= 2*t
T[N1,N1] = T[N1,N1] + UB1
T[N2,N2]= T[N2,N2] + UB2

# Energy grid
E = (np.linspace(-0.5, 5, 200))*t # Energy grid (eV)
dE = E[1] - E[0]  # Energy step
zplus = 1e-12j  # Small imaginary part to avoid singularities
VV = 0
UV = np.linspace(0, -VV, Np)

sig1 = np.zeros((Np, Np), dtype=complex)
sig2 = np.zeros((Np, Np), dtype=complex)
Tcoh = np.zeros(200)
ii = 0
                
while(ii<200):
    ck = 1 - (E[ii] + zplus - UV[0]) / (2 * t)
    ka = np.arccos(ck)
    sig1[0,0] = -t * np.exp(1j * ka)
    gam1 = 1j * (sig1 - sig1.conj().T)

    ck = 1 - (E[ii] + zplus - UV[Np-1]) / (2 * t)
    ka = np.arccos(ck)
    sig2[Np-1,Np -1] = -t * np.exp(1j * ka)
    gam2 = 1j * (sig2 - sig2.conj().T)

    G = inv((E[ii] + zplus) *np.eye(Np) - T - np.diag(UV) - sig1 - sig2)
    Tcoh[ii] = np.real(np.trace(gam1@G@gam2@(G.conj().T)))
    ii= ii+1



E=E/t
plt.plot(Tcoh, E)
plt.xlabel('Transmission function')
plt.ylabel('Energy')
plt.show()




