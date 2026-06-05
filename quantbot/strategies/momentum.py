import pandas as pd
import numpy as np
from quantbot.strategies.base import Strategy
from quantbot.features.technical import compute_rsi

class RSIMomentum(Strategy):
    """
    Momentum strategy using the Relative Strength Index.
    
    Logic:
    - Buy (1.0) when RSI crosses below oversold threshold (default 30)
    - Sell (0.0) when RSI crosses above overbought threshold (default 70)
    - Hold current position otherwise
    """
    def __init__(self, rsi_period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self._rsi_period = rsi_period
        self._oversold = oversold
        self._overbought = overbought
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        print(f"⚡ Generating signals for {self.name}...")
        
        # Calculate RSI
        features = compute_rsi(df.copy(), period=self._rsi_period)
        rsi = features['RSI']
        
        buy_condition = rsi < self._oversold
        sell_condition = rsi > self._overbought
        
        # Vectorized state logic
        events = pd.Series(np.nan, index=df.index)
        events.loc[buy_condition] = 1.0
        events.loc[sell_condition] = 0.0
        
        # Forward fill the events
        signals = events.ffill().fillna(0.0)
        
        return signals

    @property
    def name(self) -> str:
        return 'RSI Momentum'
        
    @property
    def params(self) -> dict:
        return {
            'rsi_period': self._rsi_period, 
            'oversold': self._oversold, 
            'overbought': self._overbought
        }
