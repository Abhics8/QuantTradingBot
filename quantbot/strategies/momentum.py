import pandas as pd
import numpy as np
from quantbot.strategies.base import Strategy
from quantbot.features.technical import compute_rsi


class RSIMomentum(Strategy):
    """
    Momentum (trend-following) strategy using RSI as a momentum oscillator.

    Unlike a mean-reversion RSI strategy (which fades extremes by buying
    oversold), this one RIDES strength: it goes long while RSI shows bullish
    momentum and exits when momentum fades.

    Logic:
    - Go long (1.0) when RSI rises above ``entry_level`` (momentum turning up)
    - Exit to flat (0.0) when RSI falls below ``exit_level`` (momentum fading)
    - Hold the current position in between (a hysteresis band that avoids whipsaw)
    """

    def __init__(self, rsi_period: int = 14, entry_level: float = 55.0, exit_level: float = 45.0):
        self._rsi_period = rsi_period
        self._entry_level = entry_level
        self._exit_level = exit_level

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        print(f"⚡ Generating signals for {self.name}...")

        features = compute_rsi(df.copy(), period=self._rsi_period)
        rsi = features['RSI']

        # Stateful entries/exits with hysteresis, then forward-fill the position.
        events = pd.Series(np.nan, index=df.index)
        events.loc[rsi > self._entry_level] = 1.0  # bullish momentum -> go long
        events.loc[rsi < self._exit_level] = 0.0   # momentum faded   -> go flat
        signals = events.ffill().fillna(0.0)

        return signals

    @property
    def name(self) -> str:
        return 'RSI Momentum'

    @property
    def params(self) -> dict:
        return {
            'rsi_period': self._rsi_period,
            'entry_level': self._entry_level,
            'exit_level': self._exit_level,
        }
