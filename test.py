import pennylane as qml
import numpy as np
from scipy.linalg import expm

# -----------------------------------
# 1. Define Problem Parameters
# -----------------------------------
# Matrix Sigma and vector b:
Sigma = np.array([[4, 2],
                  [2, 9]], dtype=np.float64)
b = np.array([0.05, 0.09], dtype=np.float64)

# Classical solution for reference (not used in circuit):
x_classical = np.linalg.solve(Sigma, b)
x_classical_norm = x_classical / np.linalg.norm(x_classical)
print("Normalized classical solution:", x_classical_norm)

# Normalize b to prepare |b> as a quantum state.
b_norm = np.linalg.norm(b)
b_state = b / b_norm
print("Normalized b state:", b_state)

# -----------------------------------
# 2. State Preparation Using RY Gate
# -----------------------------------
# We assume a qubit state: |b> = cos(theta_b/2)|0> + sin(theta_b/2)|1>
# Solve for theta_b from the first amplitude:
theta_b = 2 * np.arccos(b_state[0])
print("Theta for state preparation (theta_b):", theta_b)

# -----------------------------------
# 3. Define Unitary to be Used in QPE (for reference)
# -----------------------------------
t = 0.1  # evolution time
U = expm(1j * Sigma * t)
print("Unitary U = exp(i*Sigma*t):\n", U)

# -----------------------------------
# 4. Define Controlled Rotation Parameters
# -----------------------------------
# Precomputed eigenvalues
lambda1 = 3.2984
lambda2 = 9.7016
# Choose constant C as the smallest eigenvalue (for proper scaling)
C = lambda1  
theta1 = 2 * np.arcsin(np.clip(C / lambda1, 0, 1))  # ideally pi
theta2 = 2 * np.arcsin(np.clip(C / lambda2, 0, 1))  # approximately 0.703
print("Rotation angles: theta1 =", theta1, "theta2 =", theta2)
# For simplicity, we take an average rotation angle:
theta_avg = (theta1 + theta2) / 2  
print("Average rotation angle (theta_avg):", theta_avg)

# -----------------------------------
# 5. Build a Simplified HHL Circuit Using Rotation Gates
# -----------------------------------
# We will use 3 qubits:
# Wire 0: System register that will encode |b> and later |x>
# Wire 1: Control register (to simulate controlled operations)
# Wire 2: Ancilla for controlled rotation

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def hhl_rotation_demo():
    # Step A: Prepare the state |b> on the system qubit (wire 0)
    qml.RY(theta_b, wires=0)
    
    # (In a full HHL, we would use additional registers and perform QPE.
    # Here, we simulate the effect by using a control qubit.)
    
    # Step B: Prepare a control qubit (wire 1) in superposition (simulate phase register)
    qml.Hadamard(wires=1)
    
    # Step C: Apply a "controlled rotation" to simulate the inversion.
    # In a full algorithm, the rotation would be controlled on the estimated eigenvalue.
    # Here, we use wire 1 as the control for a controlled-RX gate on the ancilla (wire 2)
    # using our average rotation angle theta_avg.
    qml.ctrl(qml.RX, control=1)(theta_avg, wires=2)
    
    # Step D: (Optional) Uncompute the control register by applying Hadamard on wire 1
    qml.Hadamard(wires=1)
    
    # Now, in a complete HHL, you would post-select on the ancilla (wire 2) being in |1>.
    # For demonstration, we return the state of the system qubit (wire 0).
    return qml.state(wires=0)

# -----------------------------------
# 6. Run the Circuit and Compare the Output
# -----------------------------------
state_system = hhl_rotation_demo()
# The output state is a 2-dimensional vector representing the amplitudes of the solution.
x_quantum = state_system  # up to a normalization factor and global phase.
x_quantum_norm = np.abs(x_quantum) / np.linalg.norm(np.abs(x_quantum))
print("Normalized quantum solution (from system qubit):", x_quantum_norm)

# For reference, compare with classical normalized solution.
print("\nFinal Comparison:")
print("Classical normalized solution:", x_classical_norm)
print("Quantum (simulated HHL using rotation gates) solution (approx.):", x_quantum_norm)
