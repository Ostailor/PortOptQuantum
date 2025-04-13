import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class ClassicalPortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self):
        """Fetch historical price data and calculate returns."""
        data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(start=self.start_date, end=self.end_date)['Close']
        
        df = pd.DataFrame(data)
        self.returns = df.pct_change().dropna()
        self.mean_returns = self.returns.mean()*252
        self.cov_matrix = self.returns.cov()*252
        
        # Debug prints to inspect data
        print("Mean Returns:", self.mean_returns)
        print("Covariance Matrix:\n", self.cov_matrix)
        
    def optimize_portfolio(self, target_return=None, risk_free_rate=0.02):
        """Optimize portfolio using Markowitz mean-variance optimization."""
        n_assets = len(self.tickers)
        
        # Objective: minimize portfolio variance.
        def portfolio_variance(weights):
            return weights.T @ self.cov_matrix @ weights
        
        # Constraints: weights sum to 1.
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # If a target return is set, enforce that constraint.
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * self.mean_returns) - target_return})
        
        # Bounds: weights are between 0 and 1.
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights.
        initial_weights = np.ones(n_assets) / n_assets
        print("DEBUG: Initial guess for weights:", initial_weights)
        
        result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Debug prints: show optimization details.
        print("DEBUG: Optimization Result:")
        print("  Final Weights:", result.x)
        print("  Portfolio Variance (Objective):", portfolio_variance(result.x))
        print("  Expected Return:", np.sum(result.x * self.mean_returns))
        
        return result.x
    
    def calculate_portfolio_metrics(self, weights, risk_free_rate=0.02):
        """Calculate portfolio metrics: expected return, volatility, and Sharpe ratio."""
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_efficient_frontier(self, num_points=50, risk_free_rate=0.02):
        """
        Plot the efficient frontier and mark the Minimum Variance Portfolio (MVP).
        """
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), num_points)
        frontier_volatilities = []
        frontier_returns = []
        
        for target_return in target_returns:
            weights = self.optimize_portfolio(target_return)
            metrics = self.calculate_portfolio_metrics(weights, risk_free_rate)
            frontier_volatilities.append(metrics['volatility'])
            frontier_returns.append(metrics['expected_return'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(frontier_volatilities, frontier_returns, 'b-', label="Efficient Frontier")
        plt.xlabel("Volatility")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.grid(True)
        
        # Compute the Minimum Variance Portfolio (MVP) by optimizing without target_return.
        mvp_weights = self.optimize_portfolio()
        mvp_metrics = self.calculate_portfolio_metrics(mvp_weights, risk_free_rate)
        plt.plot(mvp_metrics['volatility'], mvp_metrics['expected_return'], 'ro', markersize=10, label="MVP")
        plt.legend()
        plt.show()
        
        return mvp_weights, mvp_metrics

def main():
    # Example usage
    tickers = ['MSFT', 'TSLA']
    end_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
    start_date = end_date - timedelta(days=365)
    
    optimizer = ClassicalPortfolioOptimizer(tickers, start_date, end_date)
    optimizer.fetch_data()
    
    # Compute the Minimum Variance Portfolio (MVP)
    mvp_weights, mvp_metrics = optimizer.plot_efficient_frontier(num_points=50)
    
    print("\nMinimum Variance Portfolio (MVP):")
    for ticker, weight in zip(tickers, mvp_weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"Expected Return: {mvp_metrics['expected_return']:.2%}")
    print(f"Volatility: {mvp_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {mvp_metrics['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    main()
