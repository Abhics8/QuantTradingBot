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
        """
        Compute the TRUE Markowitz efficient frontier: for a grid of target
        returns, solve for the minimum-variance portfolio achieving each target
        (weights sum to 1, long-only). Returns (Return, Volatility, Sharpe)
        points tracing the frontier.
        """
        df = pd.DataFrame(returns_dict).dropna()
        if df.empty:
            return pd.DataFrame()

        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        num_assets = len(df.columns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        x0 = num_assets * [1.0 / num_assets]

        def portfolio_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        targets = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
        frontier = []
        for target in targets:
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: np.sum(mean_returns * w) - t},
            )
            res = minimize(portfolio_vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if res.success:
                vol = float(res.fun)
                frontier.append({
                    'Return': float(target),
                    'Volatility': vol,
                    'Sharpe': (float(target) - self.risk_free_rate) / vol if vol > 0 else 0.0,
                })
        return pd.DataFrame(frontier)

    def random_portfolios(self, returns_dict: dict, n_portfolios: int = 5000, seed: int = 42) -> pd.DataFrame:
        """
        Monte Carlo cloud of random long-only portfolios, for plotting against
        the efficient frontier. This is NOT the frontier itself — that's
        `efficient_frontier()`.
        """
        df = pd.DataFrame(returns_dict).dropna()
        if df.empty:
            return pd.DataFrame()

        rng = np.random.default_rng(seed)
        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252
        num_assets = len(df.columns)

        rows = []
        for _ in range(n_portfolios):
            w = rng.random(num_assets)
            w /= w.sum()
            ret = float(np.sum(mean_returns * w))
            vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
            row = {'Return': ret, 'Volatility': vol,
                   'Sharpe': (ret - self.risk_free_rate) / vol if vol > 0 else 0.0}
            row.update({name: float(x) for name, x in zip(df.columns, w)})
            rows.append(row)
        return pd.DataFrame(rows)
        
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
