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
        self.cov_matrix = None
        self.mean_returns = None
        
    def fetch_data(self):
        """Fetch historical price data and calculate returns"""
        data = {}
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(start=self.start_date, end=self.end_date)['Close']
        
        df = pd.DataFrame(data)
        self.returns = df.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
    def optimize_portfolio(self, target_return=None, risk_free_rate=0.02):
        """Optimize portfolio using Markowitz mean-variance optimization"""
        n_assets = len(self.tickers)
        
        # Define objective function (minimize portfolio variance)
        def portfolio_variance(weights):
            return weights.T @ self.cov_matrix @ weights
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * self.mean_returns) - target_return
            })
        
        # Define bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio metrics"""
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def plot_efficient_frontier(self, num_points=20):
        """Plot the efficient frontier"""
        target_returns = np.linspace(
            self.mean_returns.min(),
            self.mean_returns.max(),
            num_points
        )
        
        volatilities = []
        returns = []
        
        for target_return in target_returns:
            weights = self.optimize_portfolio(target_return)
            portfolio_metrics = self.calculate_portfolio_metrics(weights)
            volatilities.append(portfolio_metrics['volatility'])
            returns.append(portfolio_metrics['expected_return'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(volatilities, returns, 'b-')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.grid(True)
        plt.show()

def main():
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    optimizer = ClassicalPortfolioOptimizer(tickers, start_date, end_date)
    optimizer.fetch_data()
    
    # Optimize for minimum variance
    min_var_weights = optimizer.optimize_portfolio()
    min_var_metrics = optimizer.calculate_portfolio_metrics(min_var_weights)
    
    print("\nMinimum Variance Portfolio:")
    for ticker, weight in zip(tickers, min_var_weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"Expected Return: {min_var_metrics['expected_return']:.2%}")
    print(f"Volatility: {min_var_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {min_var_metrics['sharpe_ratio']:.2f}")
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier()

if __name__ == "__main__":
    main() 