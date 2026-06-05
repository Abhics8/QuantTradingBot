import pandas as pd
import numpy as np

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index. Adds RSI column."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD line, Signal line, Histogram."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def compute_bollinger_bands(df: pd.DataFrame, window=20, num_std=2.0) -> pd.DataFrame:
    """Bollinger Bands. Adds BB_Upper, BB_Middle, BB_Lower, BB_Width columns."""
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * num_std)
    df['BB_Lower'] = df['BB_Middle'] - (std * num_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    return df

def compute_atr(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """Average True Range. Needs High, Low, Close. Adds ATR column."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=period).mean()
    return df

def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume. Needs Close, Volume. Adds OBV column."""
    obv = np.where(df['Close'] > df['Close'].shift(), df['Volume'],
          np.where(df['Close'] < df['Close'].shift(), -df['Volume'], 0))
    df['OBV'] = obv.cumsum()
    return df

def compute_sma(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """Simple Moving Average. Adds named SMA column."""
    df[col_name] = df['Close'].rolling(window=window).mean()
    return df

def compute_ema(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """Exponential Moving Average. Adds named EMA column."""
    df[col_name] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def compute_lagged_returns(df: pd.DataFrame, periods=[1, 5, 21]) -> pd.DataFrame:
    """Lagged percentage returns. Adds Return_1d, Return_5d, Return_21d columns."""
    for p in periods:
        df[f'Return_{p}d'] = df['Close'].pct_change(periods=p)
    return df

def compute_volatility(df: pd.DataFrame, window=21) -> pd.DataFrame:
    """Rolling volatility (std of returns). Adds Volatility_21d column."""
    returns = df['Close'].pct_change()
    df[f'Volatility_{window}d'] = returns.rolling(window=window).std()
    return df
