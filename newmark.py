
import numpy as np

def newmark_method(M, K, C, P, dof, acceleration, dt, nt):
    """
    Newmark's method for linear dynamic system.
    
    Parameters:
    M - Mass Matrix (Modal or physical)
    K - Stiffness Matrix (Modal or physical)
    C - Damping Matrix (Modal or physical)
    P - Force Matrix (Modal or physical)
    dof - system degrees of freedom
    acceleration - Type of Newmark's Method to be used ('Linear' or 'Average')
    dt - Time step
    nt - Number of time steps
    
    Returns:
    u - Displacements
    u_dot - Velocities
    u_ddot - Accelerations
    """
    
    # Choose Newmark's method parameters
    if acceleration == 'Average':
        gamma = 0.5
        beta = 0.25
    elif acceleration == 'Linear':
        gamma = 0.5
        beta = 0.1667
    else:
        raise ValueError("Invalid acceleration type. Choose 'Average' or 'Linear'.")
    
    # Constants used in Newmark's integration
    a_1 = M / (beta * dt**2) + gamma * C / (beta * dt)
    a_2 = M / (beta * dt) + (gamma / beta - 1) * C
    a_3 = (0.5 / beta - 1) * M + dt * (gamma / (2 * beta) - 1) * C
    
    # Constant multipliers for velocity and acceleration
    vel_a1 = gamma / (beta * dt)
    vel_a2 = 1 - gamma / beta
    vel_a3 = dt * (1 - gamma / (2 * beta))
    
    accl_a1 = 1 / (beta * dt**2)
    accl_a2 = 1 / (beta * dt)
    accl_a3 = 1 / (2 * beta) - 1
    
    # Initial conditions
    u = np.zeros((dof, nt))
    u_dot = np.zeros((dof, nt))
    u_ddot = np.zeros((dof, nt))
    
    u_ddot[:, 0] = np.linalg.solve(M, P[:, 0] - C @ u_dot[:, 0] - K @ u[:, 0])
    K_hat = K + a_1
    
    # Newmark integration loop
    for i in range(nt - 1):
        P_hat = P[:, i + 1] + a_1 @ u[:, i] + a_2 @ u_dot[:, i] + a_3 @ u_ddot[:, i]
        u[:, i + 1] = np.linalg.solve(K_hat, P_hat)
        u_dot[:, i + 1] = vel_a1 * (u[:, i + 1] - u[:, i]) + vel_a2 * u_dot[:, i] + vel_a3 * u_ddot[:, i]
        u_ddot[:, i + 1] = accl_a1 * (u[:, i + 1] - u[:, i]) - accl_a2 * u_dot[:, i] - accl_a3 * u_ddot[:, i]
    
    return u, u_dot, u_ddot
