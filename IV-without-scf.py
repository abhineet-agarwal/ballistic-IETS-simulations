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
mu = 0 

# Inputs
a = 3e-10  # Lattice constant (m)
t = hbar**2 / (2 * m * a**2 * q)  # Hopping parameter (eV)
beta = q * a**2 / epsil  # Scaled inverse dielectric constant
Np = 100  # Total number of sites
XX = a * 1e9 * np.arange(1, Np + 1)  # Spatial grid (nm)

# Hamiltonian matrix
T = 2 * t * np.diag(np.ones(Np)) - t * np.diag(np.ones(Np - 1), 1) - t * np.diag(np.ones(Np - 1), -1)
Jop = (q * t / ((Np-1) * hbar)) * 1j * (np.diag(np.ones(Np-1), -1) - np.diag(np.ones(Np-1), 1)) 
N1 = 50
N2 = 55
UB1= 2*t
UB2= 2*t
T[N1,N1] = T[N1,N1] + UB1
T[N2,N2]= T[N2,N2] + UB2

# Energy grid
E = (np.linspace(-0.5, 5, 200))*t # Energy grid (eV)
dE = E[1] - E[0]  # Energy step
zplus = 1e-12j  # Small imaginary part to avoid singularities
NV = 20
VV = np.linspace(0, 50, NV)
UU = np.zeros((Np, NV))
J = np.zeros((Np, NV))
VX = 0
UV = np.linspace(0, -VX, Np)


                
for kV in range(NV):
    V = VV[kV]
    print(f"V = {V}")
    sig1 = np.zeros((Np, Np), dtype=complex)
    sig2 = np.zeros((Np, Np), dtype=complex)
    ii = 0  
    rho = np.zeros((Np, Np), dtype=complex)
    f1 = n0 * np.log(1 + np.exp((mu - E/t) / kT/t))
    f2 = n0 * np.log(1 + np.exp((mu - V - E/t) / kT/t))

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
        A1 = G.conj().T @ gam1 @ G
        A2 = G.conj().T @ gam2 @ G
        rho += (dE * (f1[ii] * A1 + f2[ii] * A2) / (2 * np.pi))
        ii = ii+1
        
    UU[:, kV] = UV   
    J[:, kV] = -0.5 * q * np.diag(rho @ Jop + Jop @ rho)

E=E/t
II = np.sum(J, axis=0) 
import matplotlib.pyplot as plt
plt.figure()
plt.plot(VV, II)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('I-V Characteristic')
plt.grid(True)
plt.show()



