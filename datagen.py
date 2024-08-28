import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from newmark import newmark_method
from modal_analysis import modal_analysis
from k import get_K
from scipy.special import comb

# Parameters
dt = 0.01
T_start = 0
T_end = 20
t = np.arange(T_start, T_end + dt, dt)
nt = len(t)

# Structural Properties
dof = 8
M = 625e3 * np.eye(dof)
k = 1e9 * np.ones(dof)

# Define Force
F = np.random.randn(nt, dof).T
F[0, :] = 0


# Simulate Damage
K = get_K(5, k)['K']
alpha0 = 1e3
data = []
for i in range(dof):
    nCi = comb(dof, i+1, exact=True)
    # alphai = len((100-5)/5)*alpha0//nCi # Number of cases taken for each n-failures
    alphai = 1
    for test_case in range(alphai):
        stiffness_data = get_K(i+1, k)
        K = stiffness_data['K']
        ki = stiffness_data['k']

        # Damping
        C = 0.1 * M + 0.1 * K

        # Modal Analysis
        Mn, Kn, Cn, Fn, phi, W = modal_analysis(M, K, C, F, dof)

        # Newmark Time Integration
        acceleration = 'Linear'
        depl, vel, accl = newmark_method(M, K, C, F, dof, acceleration, dt, nt)
        data_node = {
            'k' : ki,
            'K' : K,
            'degree_of_damage': i+1,
            'degree_of_freedom' : dof,
            'depl' : depl,
            'vel' : vel,
            'accl' : accl,
            'dt': dt
        }
        data.append(data_node)

# print(data[0])
print(len(data[0]['accl'][0]))