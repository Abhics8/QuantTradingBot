import pandas as pd
import numpy as np
from quantbot.strategies.base import Strategy

class SMACrossover(Strategy):
    """
    Simple Moving Average Crossover strategy.
    
    Logic:
    - Buy (1.0) when Fast SMA > Slow SMA
    - Sell (0.0) when Fast SMA <= Slow SMA
    """
    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        self._fast_window = fast_window
        self._slow_window = slow_window
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        print(f"📈 Generating signals for {self.name}...")
        sma_fast = df['Close'].rolling(window=self._fast_window).mean()
        sma_slow = df['Close'].rolling(window=self._slow_window).mean()
        
        signals = np.where(sma_fast > sma_slow, 1.0, 0.0)
        return pd.Series(signals, index=df.index, name='Signal')

    @property
    def name(self) -> str:
        return f"SMA Crossover ({self._fast_window}/{self._slow_window})"
        
    @property
    def params(self) -> dict:
        return {'fast_window': self._fast_window, 'slow_window': self._slow_window}
