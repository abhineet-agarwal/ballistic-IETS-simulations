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
mu = 0.318  # Chemical potential (eV)
Fn = mu * np.ones(Np)  # Chemical potential array

# Initial electron density calculation
def Fhalf(x):
    xx = np.linspace(0, abs(x) + 10, 251)
    dx = xx[1] - xx[0]
    fx = (2 * dx / np.sqrt(np.pi)) * np.sqrt(xx) / (1 + np.exp(xx - x))
    y = np.sum(fx)
    return y

Nd = 2 * (n0 / 2)**1.5 * Fhalf(mu / kT)
Nd = Nd * 0.5 * np.ones(Np)

# Second derivative matrix for Poisson equation
D2 = -2 * np.diag(np.ones(Np)) + np.diag(np.ones(Np - 1), 1) + np.diag(np.ones(Np - 1), -1)
D2[0, 0] = -1
D2[-1, -1] = -1

# Hamiltonian matrix
T = 2 * t * np.diag(np.ones(Np)) - t * np.diag(np.ones(Np - 1), 1) - t * np.diag(np.ones(Np - 1), -1)
N1 = 50
N2 = 75
UB1= 0
UB2= 0
T[N1,N1] = T[N1,N1] + UB1
T[N2,N2]= T[N2,N2] + UB2

# Energy grid
E = (np.linspace(-0.5, 5, 200))*t # Energy grid (eV)
dE = E[1] - E[0]  # Energy step
zplus = 1e-12j  # Small imaginary part to avoid singularities
VV = 0
UV = np.linspace(0, -VV, Np)
f0 = n0 * np.log(1 + np.exp((mu - E) / kT))  # Fermi-Dirac distribution

U = (0.2 * np.ones(Np))
dU=0
ind=10

while(ind>0.01):
    sig1 = np.zeros((Np, Np), dtype=complex)
    sig2 = np.zeros((Np, Np), dtype=complex)
    Tcoh = np.zeros(200)
    n = np.zeros(Np)
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
        A = 1j * (G - G.conj().T)
        rhoE = f0[ii] * np.diag(A) / (2 * np.pi)
        n += (dE / a) * np.real(rhoE)
        ii= ii+1
    
    # Correction dU from Poisson
    D = np.zeros(Np)

    for k in range(Np):
        z = (Fn[k] - U[k]) / kT
        D[k] = 2 * (n0 / 2)**1.5 * (Fhalf(z + 0.1) - Fhalf(z)) / (0.1 * kT)

    dN = n - Nd + (1 / beta) * D2 @U
    dU = -beta * inv(D2 - beta * np.diag(D)) @ dN
    U += dU

    # Check for convergence
    ind = np.max(np.abs(dN)) / np.max(Nd)
    print("Convergence Indicator:", ind)


E=E/t
plt.plot(Tcoh, E)
plt.xlabel('Transmission function')
plt.ylabel('Energy')
plt.show()
