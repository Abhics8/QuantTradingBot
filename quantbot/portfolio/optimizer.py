import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Markowitz mean-variance portfolio optimization.
    """
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
        
    def optimize(self, returns_dict: dict[str, pd.Series]) -> dict:
        """
        Find optimal weights that maximize the Sharpe ratio.
        """
        # Align all returns
        df = pd.DataFrame(returns_dict).dropna()
        if df.empty:
            return {}
            
        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        num_assets = len(df.columns)
        
        def objective(weights):
            # Negative Sharpe Ratio
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_volatility
            return -sharpe
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]
        
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_weights = result.x
        opt_return = np.sum(mean_returns * opt_weights)
        opt_volatility = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
        opt_sharpe = (opt_return - self.risk_free_rate) / opt_volatility
        
        return {
            'weights': {name: w for name, w in zip(df.columns, opt_weights)},
            'expected_return': opt_return,
            'expected_volatility': opt_volatility,
            'sharpe_ratio': opt_sharpe
        }
        
    def efficient_frontier(self, returns_dict: dict, n_points: int = 50) -> pd.DataFrame:
        """Calculate the efficient frontier."""
        # A simplified approximation
        df = pd.DataFrame(returns_dict).dropna()
        if df.empty:
            return pd.DataFrame()
            
        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        num_assets = len(df.columns)
        
        results = []
        for _ in range(n_points):
            # Random portfolio generation
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_volatility
            
            res = {
                'Return': port_return,
                'Volatility': port_volatility,
                'Sharpe': sharpe
            }
            res.update({name: w for name, w in zip(df.columns, weights)})
            results.append(res)
            
        return pd.DataFrame(results)
        
    def equal_weight(self, returns_dict: dict) -> dict:
        """Simple 1/N equal weight allocation."""
        df = pd.DataFrame(returns_dict).dropna()
        if df.empty:
             return {}
        
        num_assets = len(df.columns)
        weights = {name: 1.0/num_assets for name in df.columns}
        
        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        w_arr = np.array(list(weights.values()))
        
        port_return = np.sum(mean_returns * w_arr)
        port_volatility = np.sqrt(np.dot(w_arr.T, np.dot(cov_matrix, w_arr)))
        sharpe = (port_return - self.risk_free_rate) / port_volatility
        
        return {
            'weights': weights,
            'expected_return': port_return,
            'expected_volatility': port_volatility,
            'sharpe_ratio': sharpe
        }
