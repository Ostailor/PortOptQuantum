import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import math

def is_power_of_2(n):
    """Check if a number is a power of 2."""
    return (n & (n-1) == 0) and n != 0

def pad_matrix_and_vector(A, b):
    """Pad matrix A and vector b to the next power of 2 dimensions."""
    n = A.shape[0]
    next_power_2 = 2**int(np.ceil(np.log2(n)))
    
    if n == next_power_2:
        return A, b
        
    # Pad matrix A
    A_padded = np.zeros((next_power_2, next_power_2))
    A_padded[:n, :n] = A
    
    # Pad vector b
    b_padded = np.zeros(next_power_2)
    b_padded[:n] = b
    
    return A_padded, b_padded

def solve_linear_system(A, b, learning_rate=0.01, steps=150):
    """Solve the linear system Ax = b using quantum circuits."""
    # Normalize b
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        raise ValueError("Vector b cannot be zero")
    b = b / b_norm
    
    # Make A Hermitian
    A_dag = A.conj().T
    A_bar = 0.5 * (A + A_dag)
    
    # Transform the problem
    b_bar = np.linalg.solve(A_bar, b)
    b_bar = b_bar / np.linalg.norm(b_bar)
    
    # Get number of qubits needed
    n_qubits = int(np.log2(len(b)))
    
    # Initialize quantum device
    dev = qml.device("default.qubit", wires=range(n_qubits))
    
    # Define quantum circuit
    @qml.qnode(dev)
    def circuit(params):
        # Prepare initial state
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        
        # Add entangling layers
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        return qml.state()
    
    # Define cost function
    def cost(params):
        state = circuit(params)
        exp_val = np.real(np.vdot(state, A_bar @ state))
        if abs(exp_val) < 1e-8:
            exp_val += 1e-8
        overlap = abs(np.vdot(b_bar, state))**2
        return 1 - overlap / exp_val
    
    # Initialize parameters
    params = np.random.normal(0, 0.1, n_qubits)
    
    # Optimize
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    
    print("Starting optimization:")
    for i in range(steps):
        params, cost_val = opt.step_and_cost(cost, params)
        if i % 10 == 0:
            print(f"Step {i:3d} cost = {cost_val:.6f}")
    
    print("Optimized parameters:", params)
    print("Final cost:", cost(params))
    
    # Get final state
    final_state = circuit(params)
    
    # Calculate overlap
    overlap = abs(np.vdot(b, final_state))**2
    print("Overlap (squared fidelity) between b and normalized A|psi_opt>:", overlap)
    
    return final_state

if __name__ == "__main__":
    # Test the solver with a simple example
    A = np.array([[0.8, 0.3, 0.2],
                  [0.1, 0.5, 0.3],
                  [0.2, 0.1, 0.7]])
    b = np.array([1.0, 0.0, 0.0])
    
    print("Original b =", b)
    
    # Pad if necessary
    A_padded, b_padded = pad_matrix_and_vector(A, b)
    print("Padded matrix A:")
    print(A_padded)
    print("Padded and normalized b:")
    print(b_padded)
    
    # Make A Hermitian
    A_bar = 0.5 * (A_padded + A_padded.conj().T)
    print("Transformed Hermitian operator A_bar:")
    print(A_bar)
    
    # Transform b
    b_bar = np.linalg.solve(A_bar, b_padded)
    b_bar = b_bar / np.linalg.norm(b_bar)
    print("Transformed and normalized vector b_bar =", b_bar)
    
    # Solve the system
    solution = solve_linear_system(A_padded, b_padded)
