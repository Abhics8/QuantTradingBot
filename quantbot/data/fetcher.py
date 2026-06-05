import yfinance as yf
import pandas as pd

def fetch_ohlcv(ticker: str | list[str], start_date: str, end_date: str) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Downloads historical stock data from Yahoo Finance.
    Returns OHLCV data (Open, High, Low, Close, Volume).
    """
    print(f"📡 Fetching OHLCV data for {ticker} from {start_date} to {end_date}...")
    
    # Use yfinance to download the daily data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No data returned from Yahoo Finance for {ticker}.")
        
    print("✅ Data successfully downloaded!")
    return data
