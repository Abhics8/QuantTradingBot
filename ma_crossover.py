import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance.
    """
    print(f"📡 Fetching data for {ticker} from {start_date} to {end_date}...")
    
    # Use yfinance to download the daily data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No data returned from Yahoo Finance for {ticker}.")
        
    print("✅ Data successfully downloaded!")
    return data

if __name__ == "__main__":
    # Parameters for testing our Day 2 code
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"
    
    # Call the new function
    historical_data = fetch_data(TICKER, START_DATE, END_DATE)
    
    # Print the data to verify
    print("\n--- First 5 Rows ---")
    print(historical_data.head())
    
    print("\n--- Last 5 Rows ---")
    print(historical_data.tail())
