# Quantum Portfolio Optimization

This project implements both classical and quantum approaches to portfolio optimization using the Markowitz mean-variance optimization framework. The quantum implementation uses the Variational Quantum Linear Solver (VQLS) algorithm to solve the optimization problem.

## Project Structure

- `classical_portfolio_optimizer.py`: Implements classical portfolio optimization using quadratic programming
- `quantum_portfolio_optimizer.py`: Implements quantum portfolio optimization using VQLS
- `requirements.txt`: Lists all required Python packages

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Classical Optimization
```python
from classical_portfolio_optimizer import ClassicalPortfolioOptimizer
from datetime import datetime, timedelta

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

optimizer = ClassicalPortfolioOptimizer(tickers, start_date, end_date)
optimizer.fetch_data()
weights = optimizer.optimize_portfolio()
```

### Quantum Optimization
```python
from quantum_portfolio_optimizer import QuantumPortfolioOptimizer

quantum_opt = QuantumPortfolioOptimizer(tickers, start_date, end_date)
quantum_opt.fetch_data()
quantum_weights = quantum_opt.optimize_portfolio()
```

### Comparison
Run the comparison script to see the differences between classical and quantum solutions:
```bash
python quantum_portfolio_optimizer.py
```

## Methodology

### Classical Approach
The classical implementation uses quadratic programming to solve the Markowitz mean-variance optimization problem:
- Minimizes portfolio variance subject to return constraints
- Uses the covariance matrix of asset returns
- Implements the efficient frontier

### Quantum Approach
The quantum implementation uses VQLS to solve the same optimization problem:
- Formulates the problem as a linear system of equations
- Uses a variational quantum circuit to find the solution
- Implements quantum state preparation and measurement
- Uses gradient-based optimization to minimize the cost function

## Performance Analysis

The comparison script provides:
- Portfolio weights from both approaches
- Expected returns and volatility
- Sharpe ratios
- Quantitative differences between solutions

## Industrial Application

This project demonstrates the application of quantum computing to financial portfolio optimization, which has several potential benefits:

1. **Scalability**: Quantum algorithms may handle larger portfolios more efficiently
2. **Complex Constraints**: Quantum approaches can naturally handle complex constraints
3. **Real-time Optimization**: Potential for faster optimization in real-time trading scenarios
4. **Risk Management**: Better handling of complex risk metrics and correlations

## Limitations and Future Work

1. Current quantum implementation is limited by:
   - Number of available qubits
   - Quantum noise and error rates
   - Classical-quantum interface overhead

2. Future improvements could include:
   - More sophisticated quantum ansatz
   - Error mitigation techniques
   - Hybrid quantum-classical approaches
   - Real-time data integration 