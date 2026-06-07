import numpy as np
from quantbot.backtest.engine import BacktestEngine
from quantbot.strategies.sma_crossover import SMACrossover


def test_engine_produces_equity_and_metrics(ohlcv):
    df, metrics = BacktestEngine(commission_bps=10.0, starting_capital=10_000.0).run(
        SMACrossover(fast_window=10, slow_window=30), ohlcv
    )
    assert "Strategy_Equity" in df.columns
    assert "BuyHold_Equity" in df.columns
    assert "Sharpe Ratio" in metrics
    assert df["Strategy_Equity"].iloc[-1] > 0


def test_engine_uses_lagged_signal_no_lookahead(ohlcv):
    """With zero costs, strategy return must equal YESTERDAY's signal * today's
    market return — i.e. you can't trade on information you don't have yet."""
    df, _ = BacktestEngine(commission_bps=0.0).run(
        SMACrossover(fast_window=10, slow_window=30), ohlcv
    )
    expected = (df["Signal"].shift(1) * df["Market_Return"]).fillna(0)
    actual = df["Strategy_Return"].fillna(0)
    assert np.allclose(actual, expected, atol=1e-12)
