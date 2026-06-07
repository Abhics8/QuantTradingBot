import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv():
    """Synthetic but realistic OHLCV: a positive-drift random walk, 400 days."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = np.maximum(100 + np.cumsum(rng.normal(0.05, 1.0, n)), 1.0)
    close = pd.Series(close, index=idx)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
