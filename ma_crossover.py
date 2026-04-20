import yfinance as yf
import pandas as pd
import numpy as np

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

    # Calculate the rolling mean using pandas vectorization.
    # NOTE: We intentionally DO NOT use min_periods=1. If we did, day 10's
    # "50-day SMA" would only be a 10-day average — generating premature signals
    # and inflating early performance. NaN for the first (window-1) days is correct.
    df['SMA_Short'] = df['Close'].rolling(window=window).mean()

    return df

def calculate_long_indicator(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    Day 5: Calculates the long-term Simple Moving Average (SMA).
    This mathematical indicator acts as a proxy for the overarching macroeconomic trend.
    """
    print(f"📈 Calculating {window}-day Long-Term SMA...")
    
    # Calculate the rolling mean for the Slow Window.
    # Same reasoning as Short SMA: no min_periods=1 to avoid look-ahead bias.
    df['SMA_Long'] = df['Close'].rolling(window=window).mean()
    
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day 6: Generates binary trading signals based on the Moving Average Crossover.
    1.0 = Buy / Long (Fast MA > Slow MA)
    0.0 = Sell / Flat (Fast MA <= Slow MA)
    """
    print("🚦 Generating trading signals via vectorized logic...")
    
    # Vectorized logic: np.where performs a high-speed condition check across the entire column
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1.0, 0.0)
    
    return df

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day 8: Calculates daily percentage returns for both the underlying market
    (buy-and-hold benchmark) and our crossover strategy.

    CRITICAL: We shift the Signal by 1 day before multiplying by market returns.
    The reasoning is causal — today's signal is generated from today's closing
    price, so the earliest we could actually act on it is TOMORROW's open.
    Multiplying today's signal by today's return would leak future information
    into the present, which is a classic backtesting sin.
    """
    print("💰 Calculating daily market and strategy returns...")

    # Daily percentage change of the close price = buy-and-hold return
    df['Market_Return'] = df['Close'].pct_change()

    # Strategy return = yesterday's position decision * today's realized return
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Market_Return']

    return df

def apply_transaction_costs(df: pd.DataFrame, commission_bps: float = 10.0) -> pd.DataFrame:
    """
    Day 9: Subtracts realistic transaction friction from strategy returns.

    Every time the Position column changes sign (a buy or sell execution),
    we pay a cost. We model that cost as a fixed number of basis points (bps).
    A default of 10 bps = 0.10% per trade, which is a reasonable proxy for
    retail commission + bid-ask slippage on a liquid ETF like SPY.

    Mathematically:
        cost_today = |Position_today| * (commission_bps / 10_000)
        net_return = gross_return - cost_today

    Why .abs()? Both entries (+1) and exits (-1) cost money — we never get
    paid to trade, so we take the absolute value before applying the penalty.
    """
    cost_rate = commission_bps / 10_000.0
    print(f"💸 Applying transaction costs at {commission_bps:.1f} bps per trade...")

    # Position is the diff of Signal: +1 on buys, -1 on sells, 0 when holding
    trade_cost = df['Position'].abs() * cost_rate
    df['Strategy_Return'] = df['Strategy_Return'] - trade_cost.fillna(0)

    return df

def identify_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day 7: Identifies the exact days a trade executes.
    By taking the mathematical difference (.diff()) of the Signal column:
    +1.0 = We transition from Flat to Long (Execute Buy)
    -1.0 = We transition from Long to Flat (Execute Sell)
     0.0 = We hold our current position
    """
    print("🔍 Identifying specific trade execution moments...")
    
    # Calculate the day-to-day discrete difference in the Signal column
    df['Position'] = df['Signal'].diff()
    
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
    
    # Day 5: Calculate Long-Term Indicator
    data_with_both_ma = calculate_long_indicator(data_with_short_ma, window=200)
    
    # Day 6: Generate Trading Signals
    data_with_signals = generate_signals(data_with_both_ma)
    
    # Day 7: Identify Trade Executions
    data_with_positions = identify_trades(data_with_signals)

    # Day 8: Calculate Market and Strategy Returns
    data_with_returns = calculate_returns(data_with_positions)

    # Day 9: Apply Transaction Costs (10 bps = 0.10% per trade)
    data_net_costs = apply_transaction_costs(data_with_returns, commission_bps=10.0)

    # Print the specific columns to verify the signals mathematically match the MAs
    print(f"\n--- Trade Executions Validation ({TICKER}) ---")
    print(data_net_costs[['Close', 'Signal', 'Position', 'Market_Return', 'Strategy_Return']].tail(15))
