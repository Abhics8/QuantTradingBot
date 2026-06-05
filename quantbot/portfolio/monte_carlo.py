import numpy as np
import pandas as pd

class MonteCarloSimulator:
    """
    Bootstrap Monte Carlo simulation for estimating future portfolio performance.
    """
    def __init__(self, n_simulations: int = 1000, n_days: int = 252, 
                 starting_capital: float = 10_000.0, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.n_days = n_days
        self.starting_capital = starting_capital
        np.random.seed(random_seed)
        
    def simulate(self, daily_returns: pd.Series) -> np.ndarray:
        """
        Run Monte Carlo simulation using bootstrapping (resampling with replacement).
        """
        returns_array = daily_returns.values
        
        # Generate random indices for sampling
        random_indices = np.random.randint(0, len(returns_array), 
                                           size=(self.n_simulations, self.n_days))
        
        # Get simulated daily returns
        simulated_returns = returns_array[random_indices]
        
        # Convert to growth factors (1 + r)
        growth_factors = 1 + simulated_returns
        
        # Compute cumulative products
        cumulative_growth = np.cumprod(growth_factors, axis=1)
        
        # Scale by starting capital
        equity_paths = self.starting_capital * cumulative_growth
        
        # Prepend starting capital as day 0
        starting_col = np.full((self.n_simulations, 1), self.starting_capital)
        full_paths = np.hstack((starting_col, equity_paths))
        
        return full_paths
        
    def summary_statistics(self, simulations: np.ndarray) -> dict:
        """
        Compute summary stats from simulation results.
        """
        final_values = simulations[:, -1]
        
        return {
            'median_final_value': np.median(final_values),
            'mean_final_value': np.mean(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'prob_profit': np.mean(final_values > self.starting_capital),
            'prob_double': np.mean(final_values > (self.starting_capital * 2)),
            'worst_case': np.percentile(final_values, 1),
            'best_case': np.percentile(final_values, 99)
        }
        
    def get_percentile_paths(self, simulations: np.ndarray, 
                             percentiles: list = [5, 25, 50, 75, 95]) -> pd.DataFrame:
        """
        Extract percentile paths for fan chart visualization.
        """
        paths = {}
        for p in percentiles:
            paths[f'P{p}'] = np.percentile(simulations, p, axis=0)
            
        return pd.DataFrame(paths)
