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
Ns = 15
Nc = 70
Np = Ns + Nc + Ns
XX = a * 1e9 * np.arange(1, Np + 1)  # Spatial grid (nm)
mu = 0  # Chemical potential (eV)
Fn = mu * np.ones(Np)  # Chemical potential array

# Initial electron density calculation
def Fhalf(x):
    xx = np.linspace(0, abs(x) + 10, 251)
    dx = xx[1] - xx[0]
    fx = (2 * dx / np.sqrt(np.pi)) * np.sqrt(xx) / (1 + np.exp(xx - x))
    y = np.sum(fx)
    return y

Nd = 2 * (n0 / 2)**1.5 * Fhalf(mu / kT)
Nd = Nd * np.concatenate([np.ones(Ns), 0.5*np.ones(Nc), np.ones(Ns)])


# Second derivative matrix for Poisson equation
D2 = -2 * np.diag(np.ones(Np)) + np.diag(np.ones(Np - 1), 1) + np.diag(np.ones(Np - 1), -1)
D2[0, 0] = -1
D2[-1, -1] = -1

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
VV = 0
UV = np.linspace(0, -VV, Np)
f0 = n0 * np.log(1 + np.exp((mu - E/t) / kT))  # Fermi-Dirac distribution

U = (0.2 * np.ones(Np))*t
U[N1] = 2*t
U[N2] = 2*t
dU= 0
NV = 20
VV = np.linspace(0, 50, NV)
UU = np.zeros((Np, NV))
J = np.zeros((Np, NV))


for kV in range(NV):
    V = VV[kV]
    print(f"V = {V}")
    Fn = np.concatenate([mu*np.ones(Ns), (mu-0.5*V)*np.ones(Nc), (mu-V)*np.ones(Ns)])
    ind= 10
    U = (0.2 * np.ones(Np))*t
    U[N1] = 2*t
    U[N2] = 2*t
    f1 = n0 * np.log(1 + np.exp((mu - E/t) / kT))
    f2 = n0 * np.log(1 + np.exp((mu - V - E/t) / kT))

    while(ind>0.01):
        sig1 = np.zeros((Np, Np), dtype=complex)
        sig2 = np.zeros((Np, Np), dtype=complex)
        rho = np.zeros((Np, Np), dtype=complex)
        Tcoh = np.zeros(200)
        n = np.zeros(Np)
        ii = 0
        sigs = -1j * 0.0125 * np.ones(Np)
        sigs = np.diag(sigs)
        gams = 1j * (sigs - sigs.conj().T)
        gams = np.diag(np.diag(gams))
                        
        while(ii<200):
            fs = n0 * np.log(1 + np.exp((Fn - E[N1]) / kT))
            sigin = fs * gams
            ck = 1 - (E[ii] + zplus - U[0]) / (2 * t)
            ka = np.arccos(ck)
            sig1[0,0] = -t * np.exp(1j * ka)
            gam1 = 1j * (sig1 - sig1.conj().T)

            ck = 1 - (E[ii] + zplus - U[Np-1]) / (2 * t)
            ka = np.arccos(ck)
            sig2[Np-1,Np -1] = -t * np.exp(1j * ka)
            gam2 = 1j * (sig2 - sig2.conj().T)

            G = inv((E[ii] + zplus) *np.eye(Np) - T - np.diag(U) - sig1 - sig2)
            Tcoh[ii] = np.real(np.trace(gam1@G@gam2@(G.conj().T)))
            A1 = G.conj().T @ gam1 @ G
            A2 = G.conj().T @ gam2 @ G
            rho += (dE * (f1[ii] * A1 + f2[ii] * A2) / (2 * np.pi))
            ii = ii+1
        
        D2_mod = D2.copy()
        D2_mod[N1, :] = 0
        D2_mod[N2, :] = 0
        D2_mod[N1, N1] = 1 
        D2_mod[N2, N2] = 1
        # Correction dU from Poisson
        D = np.zeros(Np)
        n = (1 / a) * np.real(np.diag(rho)) 

        for k in range(Np):
            z = (Fn[k] - U[k]) / kT
            D[k] = 2 * (n0 / 2)**1.5 * (Fhalf(z + 0.1) - Fhalf(z)) / (0.1 * kT)

        dN = n - Nd + (1 / beta) * D2_mod @U
        dN[N1] = 0
        dN[N2] = 0
        dU = -beta * inv(D2_mod - beta * np.diag(D)) @ dN
        U += dU
       # U[N1] = 2*t
       # U[N2] = 2*t
        UV = U
        # Check for convergence
        ind = np.max(np.abs(dN)) / np.max(Nd)
        #ind = np.max(np.abs(dU)) / t
        print("Convergence Indicator:", ind)
    
    UU[:, kV] = U
    J[:, kV] = -0.5 * q * np.diag(rho @ Jop + Jop @ rho)

II = np.sum(J, axis=0) 
        

E=E/t
import matplotlib.pyplot as plt
plt.figure()
plt.plot(VV, II)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('I-V Characteristic')
plt.grid(True)
plt.show()

