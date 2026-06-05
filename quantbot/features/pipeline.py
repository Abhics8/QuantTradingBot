import pandas as pd
from quantbot.features.technical import (
    compute_rsi, compute_macd, compute_bollinger_bands,
    compute_atr, compute_obv, compute_sma,
    compute_lagged_returns, compute_volatility
)

def get_feature_columns() -> list[str]:
    """Return list of all feature column names."""
    return [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
        'BB_Upper', 'BB_Lower', 'BB_Width', 'ATR', 'OBV', 
        'SMA_50', 'SMA_200', 'Return_1d', 'Return_5d', 
        'Return_21d', 'Volatility_21d'
    ]

def build_feature_matrix(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Compute all technical indicators and return clean feature matrix.
    """
    df = df.copy()
    
    # 1. Compute all indicators
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df = compute_obv(df)
    df = compute_sma(df, window=50, col_name='SMA_50')
    df = compute_sma(df, window=200, col_name='SMA_200')
    df = compute_lagged_returns(df, periods=[1, 5, 21])
    df = compute_volatility(df, window=21)
    
    # 2. Add target variable (1 if next day's return > 0 else 0)
    if include_target:
        # We want to predict tomorrow's return based on today's features
        next_day_return = df['Close'].pct_change().shift(-1)
        df['target'] = (next_day_return > 0).astype(int)
        
        # We can't predict the last row because we don't know the future
        df = df.iloc[:-1]
        
    # 3. Drop warmup rows where indicators are NaN
    # Max lookback is 200 days for SMA_200
    df = df.dropna()
    
    return df
