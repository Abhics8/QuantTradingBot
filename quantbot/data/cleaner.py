import pandas as pd

def clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Cleans and standardizes the historical dataframe by handling missing values
    and ensuring the structure is consistent for our algorithm.
    Extracts Open, High, Low, Close, Volume.
    """
    print(f"🧹 Inspecting and cleaning data for {ticker}...")
    
    # Check if the DataFrame has a MultiIndex (happens with newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # MultiIndex columns (Attribute, Ticker)
            open_prices = df['Open'][ticker]
            high_prices = df['High'][ticker]
            low_prices = df['Low'][ticker]
            close_prices = df['Close'][ticker]
            volume = df['Volume'][ticker]
        except KeyError:
            # Fallback if MultiIndex but no ticker level
            open_prices = df['Open']
            high_prices = df['High']
            low_prices = df['Low']
            close_prices = df['Close']
            volume = df['Volume']
    else:
        open_prices = df['Open']
        high_prices = df['High']
        low_prices = df['Low']
        close_prices = df['Close']
        volume = df['Volume']
        
    # Create a fresh, standardized DataFrame with OHLCV columns
    cleaned_df = pd.DataFrame(index=df.index)
    cleaned_df['Open'] = open_prices
    cleaned_df['High'] = high_prices
    cleaned_df['Low'] = low_prices
    cleaned_df['Close'] = close_prices
    cleaned_df['Volume'] = volume
    
    # Validation: Check for internal missing values (NaNs)
    missing_count = cleaned_df.isna().sum().sum()
    if missing_count > 0:
        print(f"⚠️ Found {missing_count} missing values across OHLCV. Forward-filling to clean...")
        # ffill() carries the last valid price forward, so we don't invent new prices 
        # and we don't 'look ahead' to future prices.
        cleaned_df = cleaned_df.ffill()
        
    print("✅ Data validation complete. Ready for analysis.")
    return cleaned_df
