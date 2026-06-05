import pandas as pd
import numpy as np
from quantbot.strategies.base import Strategy
from quantbot.features.technical import compute_bollinger_bands

class BollingerMeanReversion(Strategy):
    """
    Mean reversion strategy using Bollinger Bands.
    
    Logic:
    - Buy (1.0) when Close drops below the Lower Bollinger Band (oversold)
    - Sell (0.0) when Close rises above the Upper Bollinger Band (overbought)  
    - Hold current position otherwise
    """
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self._window = window
        self._num_std = num_std
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        print(f"📉 Generating signals for {self.name}...")
        
        # Calculate Bollinger Bands internally to avoid dependency on pipeline
        features = compute_bollinger_bands(df.copy(), window=self._window, num_std=self._num_std)
        
        close = features['Close']
        lower = features['BB_Lower']
        upper = features['BB_Upper']
        
        # Initialize signals
        signals = pd.Series(0.0, index=df.index, name='Signal')
        current_pos = 0.0
        
        # Iterating is slow, but stateful logic needs it. 
        # For a fast vectorized approximation:
        # We can use forward fill on signals after identifying entry/exit points
        
        buy_condition = close < lower
        sell_condition = close > upper
        
        signals.loc[buy_condition] = 1.0
        signals.loc[sell_condition] = 0.0
        
        # Forward fill to hold position until the opposite signal triggers
        # We replace 0s with NaNs only where neither condition met, then ffill
        # But initially all are 0.0. 
        
        # Better vectorized state logic:
        events = pd.Series(np.nan, index=df.index)
        events.loc[buy_condition] = 1.0
        events.loc[sell_condition] = 0.0
        
        # Forward fill the events. If it starts with NaN, fill with 0.0
        signals = events.ffill().fillna(0.0)
        
        return signals
    
    @property
    def name(self) -> str: 
        return 'Bollinger Mean Reversion'
    
    @property 
    def params(self) -> dict: 
        return {'window': self._window, 'num_std': self._num_std}
