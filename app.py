from flask import Flask, request, jsonify
from datetime import datetime
import math
import pandas as pd
import yfinance as yf
import pennylane as qml
from pennylane import numpy as np

# Import classical optimizer from its module.
from classical_portfolio_optimizer import ClassicalPortfolioOptimizer
# Import helper functions from quantum_solver module.
from quantum_solver import pad_matrix_and_vector, is_power_of_2
# Import QuantumPortfolioOptimizer from its correct module.
from quantum_portfolio_optimizer import QuantumPortfolioOptimizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/compare": {"origins": "http://localhost:3000"}})

def compare_solutions_custom(tickers, start_date, end_date, learning_rate=0.01, steps=300,
                             penalty_weight_max=5.0, extra_dist_weight_max=10.0, clip_threshold=5.0):
    """
    Compare classical and quantum portfolio optimization solutions using user-specified parameters.
    Returns a dictionary with solutions and their differences.
    """
    # Obtain the classical solution.
    classical_opt = ClassicalPortfolioOptimizer(tickers, start_date, end_date)
    classical_opt.fetch_data()
    classical_weights = classical_opt.optimize_portfolio()
    classical_metrics = classical_opt.calculate_portfolio_metrics(classical_weights)

    # Create a warm-start initial state for the quantum optimization based on classical weights.
    n_active = len(tickers)
    # The dimension in the quantum space is padded to the next power of 2 (n_active + 1 entries).
    dim = 2 ** math.ceil(math.log2(n_active + 1))
    init_state = np.zeros(dim)
    init_state[:n_active] = np.sqrt(np.array(classical_weights))
    init_state = init_state / np.linalg.norm(init_state)
    
    # Obtain the quantum solution using warm-start.
    quantum_opt = QuantumPortfolioOptimizer(tickers, start_date, end_date)
    quantum_opt.fetch_data()
    quantum_weights = quantum_opt.optimize_portfolio(
        learning_rate=learning_rate, 
        steps=steps, 
        penalty_weight_max=penalty_weight_max, 
        extra_dist_weight_max=extra_dist_weight_max,
        clip_threshold=clip_threshold,
        initial_state=init_state
    )
    quantum_metrics = quantum_opt.calculate_portfolio_metrics(quantum_weights)

    # Compute differences between classical and quantum solutions.
    weight_diff = np.linalg.norm(classical_weights - quantum_weights)
    return_diff = abs(classical_metrics['expected_return'] - quantum_metrics['expected_return'])
    vol_diff = abs(classical_metrics['volatility'] - quantum_metrics['volatility'])

    # Package results in a dictionary.
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
            "sharpe_ratio": f"{quantum_metrics['sharpe_ratio']:.2f}"
        },
        "differences": {
            "weight_difference": f"{weight_diff:.4f}",
            "return_difference": f"{return_diff:.2%}",
            "volatility_difference": f"{vol_diff:.2%}"
        }
    }
    return results

@app.route('/compare', methods=['POST', 'OPTIONS'])
@app.route('/compare', methods=['POST', 'OPTIONS'])
def compare():
    # If the request is OPTIONS, return the default preflight response.
    if request.method == 'OPTIONS':
        return app.make_default_options_response()

    # Continue processing the POST request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    tickers = data.get('tickers')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not tickers or not start_date_str or not end_date_str:
        return jsonify({"error": "tickers, start_date, and end_date are required"}), 400

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    try:
        results = compare_solutions_custom(tickers, start_date, end_date)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
