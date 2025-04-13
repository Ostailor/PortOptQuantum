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

def flatten_weights(weights):
    """
    Convert a sequence of weights into a flat 1D NumPy array of floats.
    If an element is itself a sequence, take its first element.
    """
    flat = []
    for w in weights:
        if isinstance(w, (list, np.ndarray)):
            try:
                flat.append(float(w[0]))
            except Exception as e:
                print("Error converting element:", w, "Error:", e)
        else:
            try:
                flat.append(float(w))
            except Exception as e:
                print("Error converting element:", w, "Error:", e)
    return np.array(flat)

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
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = self.cov_matrix.to_numpy()
        A[n, :n] = 1
        A[:n, n] = 1

        b = np.zeros(n + 1)
        if target_return is not None:
            b[:n] = self.mean_returns.to_numpy()
            b[n] = target_return
        else:
            b[n] = 1
        if not is_power_of_2(n + 1):
            A, b = pad_matrix_and_vector(A, b)
        return A, b

    def optimize_portfolio(
        self, target_return=None, learning_rate=0.01, steps=300,
        penalty_weight_max=5.0, extra_dist_weight_max=10.0, clip_threshold=5.0,
        initial_state=None, target_weights=None  # New parameter
    ):
        """Optimize the portfolio using VQLS with improvements.
        Returns the portfolio weights and debug logs.
        If target_weights is provided, it is used as the desired target in the cost function.
        """
        A, b = self.prepare_quantum_system(target_return)
        self.n_qubits = int(np.log2(A.shape[0]))
        self.dev = qml.device("default.qubit", wires=range(self.n_qubits))
        print(f"Optimizing on a system with {self.n_qubits} qubits. Matrix A shape: {A.shape}")

        # Define the ansatz.
        if initial_state is not None:
            print("Using provided initial state.")
            def ansatz(params):
                qml.templates.MottonenStatePreparation(initial_state, wires=range(self.n_qubits))
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=i)
            total_params = self.n_qubits
        else:
            print("No initial state provided; using default ansatz.")
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
            state = qml.state()
            # Debug: print the type and shape of state
            print("DEBUG: Inside state_circuit, state type:", type(state), "state shape:", state.shape)
            return state

        n_active = len(self.tickers)
        # Process target_weights.
        if target_weights is None:
            classical_opt = ClassicalPortfolioOptimizer(self.tickers, self.start_date, self.end_date)
            classical_opt.fetch_data()
            classical_weights = classical_opt.optimize_portfolio()
            print("Classical weights from optimizer:", classical_weights)
            # Convert classical_weights directly.
            target_weights = np.array(classical_weights, dtype=float).flatten()
        else:
            target_weights = np.array(target_weights, dtype=float).flatten()
        target_weights = target_weights / np.sum(target_weights)
        print("Final target (classical) weights:", target_weights, "with shape:", target_weights.shape)

        # Build warm-start initial state.
        if initial_state is not None:
            print("Using provided warm-start initial state.")
        else:
            print("No warm-start initial state provided; building one.")
        opt = qml.AdamOptimizer(stepsize=learning_rate)
        params = np.random.normal(0, 0.1, total_params, requires_grad=True)
        print("Initial parameters shape:", params.shape)
        debug_logs = []

        def cost(params, step):
            psi = state_circuit(params)
            #print(f"DEBUG (cost, step {step}): psi shape: {psi.shape}")
            exp_val = qml.math.sum(qml.math.conj(psi) * (A @ psi))
            exp_val = qml.math.real(exp_val)
            exp_val = qml.math.abs(exp_val) if qml.math.abs(exp_val) > 1e-8 else 1e-8
            overlap = qml.math.abs(qml.math.sum(qml.math.conj(b) * psi))**2
            base_cost = (1 - (overlap / exp_val))**2
            padded_amp = psi[n_active:]
            penalty_inactive = qml.math.sum(qml.math.abs(padded_amp)**2)
            active_amp = psi[:n_active]
            active_weights = qml.math.abs(active_amp)**2
            penalty_active = qml.math.sum((active_weights - target_weights)**2)
            p_weight = penalty_weight_max * np.tanh(step / (steps / 2))
            extra_weight = extra_dist_weight_max * np.tanh(step / (steps / 2))
            total_cost = base_cost + p_weight * penalty_inactive + extra_weight * penalty_active
            return qml.math.real(total_cost)

        for i in range(steps):
            params, cost_val = opt.step_and_cost(lambda p: cost(p, i), params)
            if i % 10 == 0:
                psi = state_circuit(params)
                #print(f"DEBUG: After step {i}, psi = {psi} and psi[:n_active]={psi[:n_active]}")
                exp_val = qml.math.sum(qml.math.conj(psi) * (A @ psi))
                exp_val = qml.math.real(exp_val)
                exp_val = qml.math.abs(exp_val) if qml.math.abs(exp_val) > 1e-8 else 1e-8
                overlap = qml.math.abs(qml.math.sum(qml.math.conj(b) * psi))**2
                base_cost = (1 - (overlap / exp_val))**2
                padded_amp = psi[n_active:]
                penalty_inactive = qml.math.sum(qml.math.abs(padded_amp)**2)
                active_amp = psi[:n_active]
                active_weights = qml.math.abs(active_amp)**2
                penalty_active = qml.math.sum((active_weights - target_weights)**2)
                p_weight = penalty_weight_max * np.tanh(i / (steps / 2))
                extra_weight = extra_dist_weight_max * np.tanh(i / (steps / 2))
                msg = (f"Step {i}: Base cost={base_cost:.4f}, Inactive penalty={penalty_inactive:.4f}, "
                       f"Active penalty={penalty_active:.4f}, Total cost={cost(params, i):.4f}")
                print(msg)
                debug_logs.append(msg)

        print("Optimized parameters:", params)
        final_cost = cost(params, steps)
        print("Final cost:", final_cost)
        final_state = state_circuit(params)
        print("DEBUG: final_state =", final_state, "with shape:", final_state.shape)
        weights = np.abs(final_state[:n_active])**2
        weights = weights / np.sum(weights)
        print("Final quantum weights:", weights)
        return weights, debug_logs

    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio metrics: expected return, volatility, and Sharpe ratio."""
        weights = np.array(weights, dtype=float)
        mean_returns = np.array(self.mean_returns, dtype=float)*252
        cov_matrix = np.array(self.cov_matrix.to_numpy(), dtype=float)*252
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
    """Compare classical and quantum portfolio optimization solutions and return a JSON-like dict."""
    tickers = ['MSFT', 'TSLA']
    end_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)

    # Classical solution.
    classical_opt = ClassicalPortfolioOptimizer(tickers, start_date, end_date)
    classical_opt.fetch_data()
    classical_weights = classical_opt.optimize_portfolio()
    classical_metrics = classical_opt.calculate_portfolio_metrics(classical_weights)

    print("DEBUG: classical_weights =", classical_weights)
    print("DEBUG: type(classical_weights) =", type(classical_weights))
    
    # Directly use classical_weights as target.
    try:
        target_weights = np.array(classical_weights, dtype=float).flatten()
    except Exception as e:
        print("DEBUG: Error converting classical_weights:", e)
    target_weights = target_weights / np.sum(target_weights)
    print("DEBUG: Target weights =", target_weights, "shape =", target_weights.shape)

    # Build warm-start initial state.
    n_active = len(tickers)
    dim = 2 ** math.ceil(math.log2(n_active + 1))
    init_state = np.zeros(dim)
    print("DEBUG: Before assignment, init_state =", init_state, "shape =", init_state.shape)
    try:
        init_state[:n_active] = np.sqrt(target_weights)
        print("DEBUG: After assignment, init_state[:n_active] =", init_state[:n_active])
    except Exception as e:
        print("DEBUG: Error in assigning init_state[:n_active]:", e)
    init_state = init_state / np.linalg.norm(init_state)
    print("DEBUG: Final init_state =", init_state, "shape =", init_state.shape)

    # Quantum solution with warm start, passing target_weights.
    quantum_opt = QuantumPortfolioOptimizer(tickers, start_date, end_date)
    quantum_opt.fetch_data()
    quantum_weights, quantum_debug = quantum_opt.optimize_portfolio(
        learning_rate=0.01, steps=300, penalty_weight_max=5.0, extra_dist_weight_max=10.0,
        clip_threshold=5.0, initial_state=init_state, target_weights=target_weights
    )
    quantum_metrics = quantum_opt.calculate_portfolio_metrics(quantum_weights)

    results = {
        "classical_solution": {
            "weights": {ticker: f"{weight:.2%}" for ticker, weight in zip(tickers, classical_weights)},
            "expected_return": f"{classical_metrics['expected_return']:.2%}",
            "volatility": f"{classical_metrics['volatility']:.2%}",
            "sharpe_ratio": f"{classical_metrics['sharpe_ratio']:.2f}"
        },
        "quantum_solution": {
            "weights": {ticker: f"{weight:.2%}" for ticker, weight in zip(tickers, quantum_weights)},
            "expected_return": f"{quantum_metrics['expected_return']:.2%}",
            "volatility": f"{quantum_metrics['volatility']:.2%}",
            "sharpe_ratio": f"{quantum_metrics['sharpe_ratio']:.2f}",
            "debug": quantum_debug
        },
        "differences": {
            "weight_difference": f"{np.linalg.norm(target_weights - quantum_weights):.4f}",
            "return_difference": f"{abs(classical_metrics['expected_return'] - quantum_metrics['expected_return']):.2%}",
            "volatility_difference": f"{abs(classical_metrics['volatility'] - quantum_metrics['volatility']):.2%}"
        }
    }
    return results

if __name__ == "__main__":
    results = compare_solutions()
    print(results)
