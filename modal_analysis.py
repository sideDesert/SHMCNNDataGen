
import numpy as np
from scipy.linalg import eig

"""
  Perform modal analysis for a dynamic system.
  
  Parameters:
  M - Mass matrix of the system
  K - Stiffness matrix of the system
  C - Damping matrix of the system
  P - Force matrix of the system
  dof - Degrees of freedom of the system
  
  Returns:
  M_m - Mass matrix in modal coordinates
  K_m - Stiffness matrix in modal coordinates
  C_m - Damping matrix in modal coordinates
  P_m - Force in modal coordinates
  Phi - Natural mode shape matrix
  Omega - Natural frequencies (rad/sec)
"""
def modal_analysis(M, K, C, P, dof):
    
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eig(K, M)
    
    # Mode shapes (sorted by eigenvalue)
    Phi = eigvecs
    
    # Natural frequencies (sorted)
    Omega_squared = np.sort(np.real(eigvals))
    Omega = np.sqrt(Omega_squared)
    
    # Modal matrices
    M_m = np.dot(Phi.T.conj(), np.dot(M, Phi))
    K_m = np.dot(Phi.T.conj(), np.dot(K, Phi))
    C_m = np.dot(Phi.T.conj(), np.dot(C, Phi))
    
    # Force in modal coordinates
    P_m = np.dot(Phi.T.conj(), P)
    
    return M_m, K_m, C_m, P_m, Phi, Omega

# Example usage
if __name__ == "__main__":
    # Define your matrices here
    M = np.array([[...]])  # Mass matrix
    K = np.array([[...]])  # Stiffness matrix
    C = np.array([[...]])  # Damping matrix
    P = np.array([[...]])  # Force matrix
    dof = len(M)           # Number of degrees of freedom
    
    M_m, K_m, C_m, P_m, Phi, Omega = modal_analysis(M, K, C, P, dof)
    
    print("Mass Matrix in Modal Coordinates:\n", M_m)
    print("Stiffness Matrix in Modal Coordinates:\n", K_m)
    print("Damping Matrix in Modal Coordinates:\n", C_m)
    print("Force Matrix in Modal Coordinates:\n", P_m)
    print("Mode Shapes:\n", Phi)
    print("Natural Frequencies:\n", Omega)
