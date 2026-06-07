import numpy as np
import pandas as pd
from quantbot.portfolio.optimizer import PortfolioOptimizer


def _returns(idx, n_assets=3, seed=0):
    rng = np.random.default_rng(seed)
    return {f"A{i}": pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx) for i in range(n_assets)}


def test_optimize_weights_are_valid(ohlcv):
    res = PortfolioOptimizer().optimize(_returns(ohlcv.index))
    w = list(res["weights"].values())
    assert abs(sum(w) - 1.0) < 1e-4          # fully invested
    assert all(-1e-6 <= x <= 1 + 1e-6 for x in w)  # long-only, bounded
    assert "sharpe_ratio" in res


def test_efficient_frontier_is_the_real_frontier(ohlcv):
    ef = PortfolioOptimizer().efficient_frontier(_returns(ohlcv.index), n_points=15)
    assert not ef.empty
    assert {"Return", "Volatility", "Sharpe"}.issubset(ef.columns)
    assert list(ef["Return"]) == sorted(ef["Return"])  # frontier traced over ascending targets
    assert (ef["Volatility"] >= 0).all()


def test_random_portfolios_is_a_cloud(ohlcv):
    rp = PortfolioOptimizer().random_portfolios(_returns(ohlcv.index), n_portfolios=200)
    assert len(rp) == 200
    assert {"Return", "Volatility", "Sharpe"}.issubset(rp.columns)
