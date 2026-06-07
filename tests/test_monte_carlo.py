import numpy as np
import pandas as pd
from quantbot.portfolio.monte_carlo import MonteCarloSimulator


def test_simulate_shape_and_stats():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252))
    mc = MonteCarloSimulator(n_simulations=200, n_days=50, starting_capital=10_000.0)
    sims = mc.simulate(r)

    assert sims.shape == (200, 51)             # n_days + 1 (day 0 = starting capital)
    assert np.allclose(sims[:, 0], 10_000.0)   # every path starts at capital

    stats = mc.summary_statistics(sims)
    for k in ["median_final_value", "prob_profit", "percentile_5", "percentile_95", "worst_case"]:
        assert k in stats
    assert 0.0 <= stats["prob_profit"] <= 1.0
