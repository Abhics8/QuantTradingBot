import pandas as pd
from quantbot.strategies.sma_crossover import SMACrossover
from quantbot.strategies.mean_reversion import BollingerMeanReversion
from quantbot.strategies.momentum import RSIMomentum


def test_signals_are_binary_and_aligned(ohlcv):
    for strat in [SMACrossover(), BollingerMeanReversion(), RSIMomentum()]:
        sig = strat.generate_signals(ohlcv)
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(ohlcv)
        assert set(float(x) for x in pd.unique(sig.dropna())).issubset({0.0, 1.0})
        assert isinstance(strat.name, str) and strat.name


def test_rsi_momentum_params_reflect_trend_following():
    p = RSIMomentum().params
    assert "entry_level" in p and "exit_level" in p   # momentum framing, not oversold/overbought
    assert p["entry_level"] > p["exit_level"]          # hysteresis band
