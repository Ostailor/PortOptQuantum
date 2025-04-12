import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import math
from classical_portfolio_optimizer import ClassicalPortfolioOptimizer
from quantum_solver import pad_matrix_and_vector, is_power_of_2

def clip_gradients(grads, clip_value=5.0):
    """Clip each gradient component to the range [-clip_value, clip_value]."""
    return np.clip(grads, -clip_value, clip_value)

class QuantumPortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None
        self.n_qubits = None
        self.dev = None
        
    def fetch_data(self):
        """Fetch historical price data and calculate returns."""
        data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(start=self.start_date, end=self.end_date)['Close']
        
        df = pd.DataFrame(data)
        self.returns = df.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def prepare_quantum_system(self, target_return=None):
        """Prepare the quantum system for VQLS."""
        n = len(self.tickers)
        
        # Build the constraint matrix (active subspace from covariance).
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = self.cov_matrix
        A[n, :n] = 1
        A[:n, n] = 1
        
        # Build the right-hand side vector.
        b = np.zeros(n + 1)
        if target_return is not None:
            b[:n] = self.mean_returns
            b[n] = target_return
        else:
            b[n] = 1  # Normalization constraint.
            
        # Pad matrix and vector to power-of-2 dimensions if needed.
        if not is_power_of_2(n + 1):
            A, b = pad_matrix_and_vector(A, b)
            
        return A, b
    
    def optimize_portfolio(
        self, target_return=None, learning_rate=0.01, steps=300, 
        penalty_weight_max=5.0, extra_dist_weight_max=10.0, clip_threshold=5.0,
        initial_state=None
    ):
        """Optimize the portfolio using VQLS with improvements.
        
        When initial_state is provided (as a vector for the full Hilbert space),
        the ansatz will first use MottonenStatePreparation to warm start the circuit.
        """
        A, b = self.prepare_quantum_system(target_return)
        
        # Determine number of qubits.
        self.n_qubits = int(np.log2(A.shape[0]))
        self.dev = qml.device("default.qubit", wires=range(self.n_qubits))
        
        # If an initial state is given, use it in a warm-start block.
        if initial_state is not None:
            def ansatz(params):
                # Warm start: prepare the state corresponding to classical solution.
                qml.templates.MottonenStatePreparation(initial_state, wires=range(self.n_qubits))
                # Refinement block: apply a single layer of parameterized RY gates.
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=i)
            total_params = self.n_qubits
        else:
            # Otherwise, use a two-layer generic ansatz.
            def ansatz(params):
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=i)
                if self.n_qubits > 1:
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                for i in range(self.n_qubits):
                    qml.RY(params[i + self.n_qubits], wires=i)
            total_params = 2 * self.n_qubits
        
        @qml.qnode(self.dev, interface="autograd")
        def state_circuit(params):
            ansatz(params)
            return qml.state()
        
        n_active = len(self.tickers)
        # Target distribution for the active subspace: uniform distribution.
        target_weights = np.ones(n_active) / n_active
        
        # Use an Adam optimizer.
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        params = np.random.normal(0, 0.1, total_params, requires_grad=True)
        
        def cost(params, step):
            psi = state_circuit(params)
            # Base cost: enforce that A|psi> approximates b.
            exp_val = qml.math.sum(qml.math.conj(psi) * (A @ psi))
            exp_val = qml.math.real(exp_val)
            exp_val = exp_val if qml.math.abs(exp_val) > 1e-8 else exp_val + 1e-8
            overlap = qml.math.abs(qml.math.sum(qml.math.conj(b) * psi))**2
            base_cost = 1 - overlap / exp_val
            
            # Penalty for amplitude in the padded (inactive) subspace.
            padded_amp = psi[n_active:]
            penalty_inactive = qml.math.sum(qml.math.abs(padded_amp)**2)
            
            # Penalty for active weights deviating from target.
            active_amp = psi[:n_active]
            active_weights = qml.math.abs(active_amp)**2
            penalty_active = qml.math.sum((active_weights - target_weights)**2)
            
            # Anneal penalty weights linearly over iterations.
            p_weight = (step / steps) * penalty_weight_max
            extra_weight = (step / steps) * extra_dist_weight_max
            
            total_cost = base_cost + p_weight * penalty_inactive + extra_weight * penalty_active
            return qml.math.real(total_cost)
        
        # Optimization loop with gradient clipping.
        for i in range(steps):
            params, cost_val = opt.step_and_cost(lambda p: cost(p, i), params)
            # Compute gradients and clip them.
            grad = qml.grad(lambda p: cost(p, i))(params)
            params = params - learning_rate * clip_gradients(grad, clip_threshold)
            
            if i % 10 == 0:
                print(f"Step {i}: Cost = {cost(params, i):.4f}")
        
        print("Optimized parameters:", params)
        final_cost = cost(params, steps)
        print("Final cost:", final_cost)
        
        final_state = state_circuit(params)
        weights = np.abs(final_state[:n_active])**2
        weights = weights / np.sum(weights)
        
        return weights
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio metrics: expected return, volatility, and Sharpe ratio."""
        weights = np.array(weights, dtype=float)
        mean_returns = np.array(self.mean_returns, dtype=float)
        cov_matrix = np.array(self.cov_matrix.to_numpy(), dtype=float)
        
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

def compare_solutions():
    """Compare classical and quantum portfolio optimization solutions."""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)
    
    # Get classical solution.
    classical_opt = ClassicalPortfolioOptimizer(tickers, start_date, end_date)
    classical_opt.fetch_data()
    classical_weights = classical_opt.optimize_portfolio()
    classical_metrics = classical_opt.calculate_portfolio_metrics(classical_weights)
    
    # Now build a warm-start initial state from the classical solution.
    # Classical weights correspond to the active subspace (length = number of tickers).
    n_active = len(tickers)
    # The underlying padded space is determined by prepare_quantum_system: dimension = n+1 padded to power-of-2.
    dim = 2 ** math.ceil(math.log2(n_active + 1))
    # Create a full state vector: first n_active entries are sqrt(classical weight), rest zeros.
    init_state = np.zeros(dim)
    init_state[:n_active] = np.sqrt(np.array(classical_weights))
    # Normalize (should be nearly normalized already, but for safety).
    init_state = init_state / np.linalg.norm(init_state)
    
    # Quantum solution with warm start.
    quantum_opt = QuantumPortfolioOptimizer(tickers, start_date, end_date)
    quantum_opt.fetch_data()
    quantum_weights = quantum_opt.optimize_portfolio(
        learning_rate=0.01, steps=300, penalty_weight_max=5.0, extra_dist_weight_max=10.0,
        clip_threshold=5.0, initial_state=init_state
    )
    quantum_metrics = quantum_opt.calculate_portfolio_metrics(quantum_weights)
    
    print("\nClassical Solution:")
    for ticker, weight in zip(tickers, classical_weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"Expected Return: {classical_metrics['expected_return']:.2%}")
    print(f"Volatility: {classical_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {classical_metrics['sharpe_ratio']:.2f}")
    
    print("\nQuantum Solution:")
    for ticker, weight in zip(tickers, quantum_weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"Expected Return: {quantum_metrics['expected_return']:.2%}")
    print(f"Volatility: {quantum_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {quantum_metrics['sharpe_ratio']:.2f}")
    
    weight_diff = np.linalg.norm(classical_weights - quantum_weights)
    return_diff = abs(classical_metrics['expected_return'] - quantum_metrics['expected_return'])
    vol_diff = abs(classical_metrics['volatility'] - quantum_metrics['volatility'])
    
    print("\nDifferences:")
    print(f"Weight difference (L2 norm): {weight_diff:.4f}")
    print(f"Return difference: {return_diff:.2%}")
    print(f"Volatility difference: {vol_diff:.2%}")

if __name__ == "__main__":
    compare_solutions()
