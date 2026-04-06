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

def clean_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Cleans and standardizes the historical dataframe by handling missing values
    and ensuring the structure is consistent for our algorithm.
    """
    print("🧹 Inspecting and cleaning data...")
    
    # Check if the DataFrame has a MultiIndex (happens with newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        close_prices = df['Close'][ticker]
    else:
        close_prices = df['Close']
        
    # Create a fresh, standardized DataFrame with only the Close column
    cleaned_df = pd.DataFrame(index=df.index)
    cleaned_df['Close'] = close_prices
    
    # Validation: Check for internal missing values (NaNs)
    missing_count = cleaned_df['Close'].isna().sum()
    if missing_count > 0:
        print(f"⚠️ Found {missing_count} missing values. Forward-filling to clean...")
        # ffill() carries the last valid price forward, so we don't invent new prices 
        # and we don't 'look ahead' to future prices.
        cleaned_df['Close'] = cleaned_df['Close'].ffill()
        
    print("✅ Data validation complete. Ready for analysis.")
    return cleaned_df

def calculate_short_indicator(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Day 4: Calculates the short-term Simple Moving Average (SMA).
    This mathematical indicator acts as a proxy for short-term market momentum.
    """
    print(f"📈 Calculating {window}-day Short-Term SMA...")
    
    # Calculate the rolling mean using pandas vectorization
    # min_periods=1 ensures we get values even before day 50 (e.g., day 10 is a 10-day average)
    df['SMA_Short'] = df['Close'].rolling(window=window, min_periods=1).mean()
    
    return df

if __name__ == "__main__":
    # Parameters for testing our Day 2 code
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"
    
    # Fetch raw data
    raw_data = fetch_data(TICKER, START_DATE, END_DATE)
    
    # Validate and clean the data
    ready_data = clean_data(raw_data, TICKER)
    
    # Day 4: Calculate Short-Term Indicator
    data_with_short_ma = calculate_short_indicator(ready_data, window=50)
    
    # Print the specific columns to verify our math
    print(f"\n--- Short-Term MA Validation ({TICKER}) ---")
    print(data_with_short_ma[['Close', 'SMA_Short']].tail(10))
