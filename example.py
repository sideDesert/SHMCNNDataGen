import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from newmark import newmark_method
from modal_analysis import modal_analysis
from k import get_K

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

# Simulate Damage
K = get_K(5, k)['K']

# Define Force
F = np.random.randn(nt, dof).T
F[0, :] = 0

# Damping
C = 0.1 * M + 0.1 * K

# Modal Analysis
Mn, Kn, Cn, Fn, phi, W = modal_analysis(M, K, C, F, dof)


# Newmark Time Integration
acceleration = 'Linear'
depl_PC, vel_PC, accl_PC = newmark_method(M, K, C, F, dof, acceleration, dt, nt)

# Plot results
fig, axs = plt.subplots(dof // 2, dof // 4, figsize=(15, 10))
for i in range(dof):
    ax = axs[i // (dof // 4), i % (dof // 4)]
    ax.plot(t, depl_PC[i, :], '--', label='PC')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement')
    ax.legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(dof // 2, dof // 4, figsize=(15, 10))
for i in range(dof):
    ax = axs[i // (dof // 4), i % (dof // 4)]
    ax.plot(t, vel_PC[i, :], '--', label='PC')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(dof // 2, dof // 4, figsize=(15, 10))
for i in range(dof):
    ax = axs[i // (dof // 4), i % (dof // 4)]
    ax.plot(t, accl_PC[i, :], '--', label='PC')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration')
    ax.legend()

plt.tight_layout()
plt.show()
