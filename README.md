# Quantum Portfolio Optimization

This project demonstrates a hybrid quantum-classical approach to portfolio optimization using a Variational Quantum Linear Solver (VQLS) inspired by the Harrow–Hassidim–Lloyd (HHL) algorithm. In our implementation, we compare classical portfolio optimization with a quantum variational algorithm using Pennylane.

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

Portfolio optimization aims to determine the optimal allocation of assets to maximize return for a given level of risk. The classical Markowitz mean–variance model formulates this as a quadratic optimization problem. In contrast, our quantum approach employs a variational quantum linear solver (VQLS) that leverages ideas from the HHL algorithm to solve linear systems of equations. This project compares the portfolio weights, expected returns, volatility, and Sharpe ratios computed by both classical and quantum methods.

## Mathematical Background

### Classical Portfolio Optimization

In the classical framework, we formulate the problem as follows:

$$
\min_{\mathbf{w}} \quad \frac{1}{2}\,\mathbf{w}^T \Sigma \, \mathbf{w}
$$
subject to
$$
\mathbf{r}^T \mathbf{w} = R_{\text{target}}, \quad \text{and} \quad \mathbf{1}^T \mathbf{w} = 1,
$$

where:
- $\Sigma$ is the covariance matrix of asset returns.
- $\mathbf{r}$ is the vector of expected asset returns.
- $R_{\text{target}}$ is the target return.

The solution can be obtained by solving a linear system and is often computed using standard numerical methods such as quadratic programming.

### Quantum Portfolio Optimization (VQLS/HHL Approach)

Our quantum algorithm aims to solve a linear system of the form

$$
Ax = b,
$$

where:
- $A$ is an augmented matrix derived from the covariance matrix and constraint conditions.
- $b$ is a vector constructed from asset returns and budget constraints.

The HHL algorithm theoretically computes the solution $x = A^{-1}b$ by:
1. **Preparing a quantum state** $|b\rangle$ that encodes the vector $b$.
2. **Performing quantum phase estimation (QPE)** on a unitary $U = e^{iAt}$ to implicitly determine the eigenvalues $\lambda_j$ and eigenvectors $|u_j\rangle$ of $A$.  
   In the eigenbasis, we express:
   $
   |b\rangle = \sum_{j} \beta_j \, |u_j\rangle,
   $
   so that
   $
   A^{-1}|b\rangle = \sum_{j} \frac{\beta_j}{\lambda_j} |u_j\rangle.
   $
3. **Controlled rotations** are applied to "load" the reciprocal eigenvalues $1/\lambda_j$ into an ancilla qubit, effectively performing the inversion.
4. **Uncomputation and post-selection** remove the ancillary data, leaving a state proportional to the solution $x$.

In our variational (VQLS) approach, we simulate this behavior by designing an ansatz, defining a cost function that penalizes deviations from the classical target, and then using a classical optimizer in a hybrid loop.

## Implementation Details

### Project Structure

- **classical_portfolio_optimizer.py:**  
  Contains the classical portfolio optimization routines and functions for computing portfolio weights, expected returns, volatility, and Sharpe ratio.

- **quantum_solver.py:**  
  Provides helper functions such as `pad_matrix_and_vector` and `is_power_of_2`, which are used to prepare the augmented matrix $A$ and vector $b$ for the quantum algorithm.

- **quantum_portfolio_optimizer.py:**  
  Contains the `QuantumPortfolioOptimizer` class. This class implements the variational quantum linear solver (VQLS) using Pennylane. It defines the ansatz, cost function, and optimization loop.

- **app.py:**  
  A Flask application that exposes an API endpoint (`/compare`) to compare classical and quantum portfolio solutions. It utilizes the modules above to fetch data, run optimizations, and output results in JSON format.

### Key Modules and Functions

- **Ansatz and Cost Function (quantum_portfolio_optimizer.py):**  
  - **Ansatz:** Constructs a quantum circuit using single-qubit rotations (e.g., $R_Y$ gates) and entangling gates (e.g., CNOTs).
  - **Cost Function:** The cost is a combination of:
    - A base cost that measures the difference between the state obtained via the quantum circuit and the state \(b\).
    - Penalty terms that force the "active" portion of the state (corresponding to assets) to be close to the target weights.
    
- **Warm-Start Initial State:**  
  The warm-start is constructed by taking the square root of the classical weights. This state is used as an initial guess for the quantum optimizer.

## Performance Evaluation and Analysis

### Accuracy

- **Comparison Metrics:**  
  We compare portfolio weights, expected returns, volatility, and Sharpe ratios between the classical and quantum solutions.
- **Observed Results:**  
  In our experiments with a two-asset portfolio (e.g., "MSFT" and "TSLA"), the classical optimizer might yield weights like `[1.0, 0.0]`. The quantum optimizer, guided by a cost function that penalizes deviations from this target, converges to weights close to `[0.9895, 0.0105]`—indicating high accuracy in matching the classical solution.

### Resource Requirements

- **Qubit Count:**  
  Our example uses an augmented $4 \times 4$ matrix for two assets, which is implemented using $\log_2(4)=2$ qubits.
- **Quantum Gates:**  
  The ansatz consists of a small number of $R_Y$ gates and CNOT gates. Resource requirements will increase with more assets.
- **Hybrid Loop:**  
  The algorithm relies on a classical optimizer (e.g., Adam) to update the quantum circuit parameters.

### Scalability

- **Quantum Advantage:**  
  The HHL algorithm offers exponential speedup in solving linear systems under ideal conditions (sparse matrices, low condition number).  
- **Variational Approach:**  
  Our VQLS method is better suited for near-term devices (NISQ era) but may require careful ansatz design and noise management to scale to larger problems.
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
     Then, send a POST request to `http://localhost:5000/compare` with JSON containing:
     ```json
     {
         "tickers": ["MSFT", "TSLA"],
         "start_date": "2020-01-01",
         "end_date": "2021-01-01"
     }
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
