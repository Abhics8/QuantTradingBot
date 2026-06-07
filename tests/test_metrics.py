import numpy as np
import pandas as pd
from quantbot.backtest.metrics import compute_metrics


def test_max_drawdown_is_exact():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    equity = pd.Series([100.0, 110.0, 99.0, 120.0], index=idx)  # peak 110 -> trough 99
    ret = equity.pct_change().fillna(0.0)
    df = pd.DataFrame(
        {"Strategy_Return": ret, "Strategy_Equity": equity, "Signal": 1.0, "Position": 0.0},
        index=idx,
    )
    m = compute_metrics(df)
    assert abs(m["Max Drawdown (%)"] - (-10.0)) < 1e-6  # (99-110)/110 = -10%


def test_metrics_keys_and_bounds(ohlcv):
    idx = ohlcv.index
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    equity = 10_000 * (1 + ret).cumprod()
    df = pd.DataFrame(
        {"Strategy_Return": ret, "Strategy_Equity": equity, "Signal": 1.0, "Position": 0.0},
        index=idx,
    )
    m = compute_metrics(df)
    for k in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "CAGR (%)", "Max Drawdown (%)"]:
        assert k in m and isinstance(m[k], float)
    assert m["Max Drawdown (%)"] <= 0
    assert m["Annualized Volatility (%)"] >= 0


def test_empty_returns_safe():
    df = pd.DataFrame({"Strategy_Return": [], "Strategy_Equity": []})
    assert compute_metrics(df) == {}
