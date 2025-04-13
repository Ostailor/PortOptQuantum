# Quantum Portfolio Optimization

This project demonstrates a hybrid quantum-classical approach to portfolio optimization using a variational quantum linear solver (VQLS) inspired by the Harrow–Hassidim–Lloyd (HHL) algorithm. It compares the classical portfolio optimization solution with a quantum solution implemented with Pennylane.

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
  - [Classical Portfolio Optimization](#classical-portfolio-optimization)
  - [Quantum Portfolio Optimization (VQLS/HHL Approach)](#quantum-portfolio-optimization-vqlshhl-approach)
- [Implementation Details](#implementation-details)
  - [Project Structure](#project-structure)
  - [Key Modules and Functions](#key-modules-and-functions)
- [Performance Evaluation and Analysis](#performance-evaluation-and-analysis)
  - [Accuracy](#accuracy)
  - [Resource Requirements](#resource-requirements)
  - [Scalability](#scalability)
- [Reproducing the Results](#reproducing-the-results)
- [Future Work and Improvements](#future-work-and-improvements)
- [License](#license)

## Overview

Portfolio optimization involves allocating assets to maximize return for a given risk level. The classical Markowitz mean–variance model does this by minimizing the portfolio variance under certain return and budget constraints. My quantum variant applies a hybrid quantum-classical approach using a variational quantum linear solver (VQLS) to solve the underlying linear system.

## Mathematical Background

### Classical Portfolio Optimization

In the classical approach, we solve the following optimization problem:

$$
\min_{\mathbf{w}} \mathbf{w}^T \Sigma  \mathbf{w}
$$

subject to

$$
\quad \mathbf{1}^T \mathbf{w} = 1,
$$

where:
- $\Sigma$ is the covariance matrix of the asset returns,
- $\mathbf{w}$ is the weights of each asset, and
- $\mathbf{1}$ is the vector of ones.

The solution is computed using standard classical optimization techniques.

### Quantum Portfolio Optimization (VQLS/HHL Approach)

My quantum algorithm addresses the problem by solving the linear system

$$
A x = b,
$$

where $A$ is an augmented matrix constructed from $\Sigma$ and the constraints, and $b$ is the corresponding vector.

The algorithm is based on the spectral decomposition of $A$. First, the vector $b$ is encoded into a quantum state $|b\rangle$, and we express it in the eigenbasis of $A$:

$$
|b\rangle = \sum_{j} \beta_j \, |u_j\rangle,
$$

where

$$
|b\rangle = \sum_{j} \beta_j \, |u_j\rangle,
$$

Then, the solution state (proportional to $x = A^{-1}b$) is given by

$$
A^{-1}|b\rangle = \sum_{j} \frac{\beta_j}{\lambda_j} |u_j\rangle,
$$

where $\lambda_j$ are the eigenvalues of $A$. In my variational approach, we design an ansatz, define a cost function that penalizes deviation from the desired state, and use a classical optimizer to adjust the parameters of the quantum circuit.

## Implementation Details

### Project Structure

- **classical_portfolio_optimizer.py:** Contains classical portfolio optimization routines.
- **quantum_solver.py:** Provides helper functions (e.g., `pad_matrix_and_vector`, `is_power_of_2`) for preparing the matrix $A$ and vector $b$.
- **quantum_portfolio_optimizer.py:** Implements the `QuantumPortfolioOptimizer` class, which defines the variational ansatz, the cost function, and the optimization routine using Pennylane.
- **app.py:** A Flask application that exposes an API endpoint (`/compare`) to compare classical and quantum solutions.

### Key Modules and Functions

- **Ansatz and Cost Function:**  
  The quantum ansatz uses rotation gates (such as $R_Y$) and CNOTs to prepare a candidate quantum state. The cost function measures the difference between the quantum state and the target state (which is the classical solution, if specified), including penalty terms for deviations in the "active" subspace corresponding to asset weights.

- **Warm-Start Initial State:**  
  The warm-start initial state is built using the square roots of the classical weights, providing a good initial guess for the variational optimization.

## Performance Evaluation and Analysis

### Accuracy

- **Comparison Metrics:**  
  We compare portfolio weights, expected returns, volatility, and Sharpe ratios between the classical and quantum solutions.
- **Observed Results:**  
  In my experiments with a two-asset portfolio (e.g., "MSFT" and "TSLA"), the classical optimizer might yield weights like `[1.0, 0.0]`. The quantum optimizer, guided by a cost function that penalizes deviations from this target, converges to weights close to `[0.9895, 0.0105]`—indicating high accuracy in matching the classical solution.

### Resource Requirements

- **Qubit Count:**  
  My example uses an augmented $4 \times 4$ matrix for two assets, which is implemented using $\log_2(4)=2$ qubits.
- **Quantum Gates:**  
  The ansatz consists of a small number of $R_Y$ gates and CNOT gates. Resource requirements will increase with more assets.
- **Hybrid Loop:**  
  The algorithm relies on a classical optimizer (e.g., Adam) to update the quantum circuit parameters.

### Scalability

- **Quantum Advantage:**  
  The HHL algorithm offers exponential speedup in solving linear systems under ideal conditions (sparse matrices, low condition number).  
- **Variational Approach:**  
  My VQLS method is better suited for near-term devices (NISQ era) but requires careful ansatz design and noise management to scale to larger problems.
- **Hybrid Complexity:**  
  As problem size increases, both the quantum circuit depth and the number of variational parameters increase.

## Reproducing the Results

1. **Environment Setup:**
   - **Python Version:** Python 3.8+ is recommended.
   - **Dependencies:**  
     Install required packages using pip:
     ```bash
     pip install -r requirements.txt
     ```
2. **Project Files:**
   - Ensure the following modules are available:
     - `classical_portfolio_optimizer.py`
     - `quantum_solver.py`
     - `quantum_portfolio_optimizer.py`
     - `app.py`
3. **Running the Application:**
   - **Flask API:**  
     To start the web API, run:
     ```bash
     python app.py
     ```
     Then to run the front end:
     ```bash
     cd portfolio-optimizer-frontend
     npm run build
     serve -s build
     ```
   - **Direct Execution:**  
     You can also run the comparison functions directly from a script to see console output with debug information.

## Future Work and Improvements

- **Enhanced Ansatz:**  
  Experiment with different ansatz designs to reduce circuit depth and improve convergence.
- **Noise Mitigation:**  
  Develop techniques to account for hardware noise as the quantum hardware scales up.
- **Scalability Analysis:**  
  Extend experiments to portfolios with more assets to analyze how qubit count and circuit complexity grow.
- **Hybrid Strategies:**  
  Integrate more sophisticated classical optimizers or use iterative feedback to guide parameter updates.

## License

[MIT License](LICENSE)

---

This README, along with the well-documented source code, provides a comprehensive overview of the project, the underlying mathematics of both classical and quantum optimization, and instructions to reproduce the results. Feel free to modify this README to better reflect any additional details or future enhancements.
