import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io

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

def plot_results(df: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """
    Day 12: Renders a two-panel matplotlib chart summarizing the backtest.

    Panel 1 (top):    Price with Short/Long SMA overlays and buy/sell markers
                      on the exact days the strategy transitioned position.
    Panel 2 (bottom): Strategy equity curve vs. Buy-and-Hold benchmark so
                      outperformance (or underperformance) is visually obvious.
    """
    print("🎨 Rendering backtest visualization...")

    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
    )

    # --- Top panel: price + moving averages + trade markers ---
    ax_price.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.2)
    ax_price.plot(df.index, df['SMA_Short'], label='50-day SMA', color='tab:blue', linewidth=1.0)
    ax_price.plot(df.index, df['SMA_Long'], label='200-day SMA', color='tab:orange', linewidth=1.0)

    # Buy = Position +1 (Flat -> Long), Sell = Position -1 (Long -> Flat)
    buy_days = df[df['Position'] == 1.0]
    sell_days = df[df['Position'] == -1.0]
    ax_price.scatter(buy_days.index, buy_days['Close'],
                     marker='^', color='green', s=110, label='Buy (Golden Cross)', zorder=5)
    ax_price.scatter(sell_days.index, sell_days['Close'],
                     marker='v', color='red', s=110, label='Sell (Death Cross)', zorder=5)

    ax_price.set_title(f'{ticker} — Moving Average Crossover Strategy', fontsize=14, fontweight='bold')
    ax_price.set_ylabel('Price (USD)')
    ax_price.legend(loc='upper left')
    ax_price.grid(alpha=0.3)

    # --- Bottom panel: equity curves ---
    ax_equity.plot(df.index, df['Strategy_Equity'],
                   label='Strategy (net of costs)', color='tab:green', linewidth=1.6)
    ax_equity.plot(df.index, df['BuyHold_Equity'],
                   label='Buy & Hold benchmark', color='tab:gray', linewidth=1.2, linestyle='--')
    ax_equity.set_title('Equity Curve Comparison', fontsize=12)
    ax_equity.set_ylabel('Portfolio Value (USD)')
    ax_equity.set_xlabel('Date')
    ax_equity.legend(loc='upper left')
    ax_equity.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"✅ Chart saved to {save_path}")
    plt.show()

def compute_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Day 11: Computes the professional backtest tear-sheet metrics.

    - Total Return (%): simple percentage growth of the final equity over start.
    - CAGR (%): annualized compounded growth rate — lets us compare strategies
      run over different time spans on equal footing.
    - Annualized Volatility (%): std-dev of daily returns scaled by sqrt(252),
      the standard approximation for 252 trading days per year.
    - Sharpe Ratio: (mean excess return / std dev) annualized. The industry
      benchmark for risk-adjusted return. >1 is decent, >2 is strong.
    - Max Drawdown (%): the worst peak-to-trough decline ever experienced.
      This is often the metric that actually determines if a strategy is
      psychologically trade-able — big drawdowns shake investors out.
    """
    print("📐 Computing performance tear-sheet metrics...")

    strat_returns = df['Strategy_Return'].dropna()
    strat_equity = df['Strategy_Equity'].dropna()

    # --- Total + Annualized Return ---
    total_return = strat_equity.iloc[-1] / strat_equity.iloc[0] - 1
    years = len(strat_returns) / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # --- Volatility ---
    ann_vol = strat_returns.std() * np.sqrt(252)

    # --- Sharpe Ratio ---
    # Daily risk-free rate = annual rate / 252
    daily_rf = risk_free_rate / 252.0
    excess_returns = strat_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0

    # --- Max Drawdown ---
    # Running peak of equity curve, then percent decline below that peak.
    running_peak = strat_equity.cummax()
    drawdown_series = (strat_equity - running_peak) / running_peak
    max_drawdown = drawdown_series.min()

    metrics = {
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr * 100,
        'Annualized Volatility (%)': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown * 100,
    }
    return metrics

def print_metrics(metrics: dict) -> None:
    """Pretty-prints the metrics dictionary as a readable tear-sheet."""
    print("\n" + "=" * 42)
    print("     STRATEGY PERFORMANCE TEAR-SHEET")
    print("=" * 42)
    for name, value in metrics.items():
        print(f"  {name:<28}{value:>10.2f}")
    print("=" * 42)

def build_equity_curve(df: pd.DataFrame, starting_capital: float = 10_000.0) -> pd.DataFrame:
    """
    Day 10: Compounds daily returns into a running portfolio value ("equity curve").

    The math is a classic geometric compounding chain:
        equity_t = starting_capital * Π (1 + return_i)  for i = 1..t

    Pandas expresses this in one line via (1 + r).cumprod().

    We build TWO curves so the reader can compare them side-by-side:
        1. Strategy_Equity  — our crossover bot, net of transaction costs
        2. BuyHold_Equity   — a passive baseline that just holds the index

    The hard truth of quant backtesting: most simple strategies fail to
    beat buy-and-hold once realistic frictions are applied. This curve
    is what makes that verdict visible.
    """
    print(f"📊 Compounding equity curve from ${starting_capital:,.0f} starting capital...")

    # fillna(0) is safe here — early rows have NaN returns (before SMA warmup)
    # and treating those days as flat (0% return) keeps the product chain valid.
    strategy_growth = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    benchmark_growth = (1 + df['Market_Return'].fillna(0)).cumprod()

    df['Strategy_Equity'] = starting_capital * strategy_growth
    df['BuyHold_Equity'] = starting_capital * benchmark_growth

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

def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    fast_window: int = 50,
    slow_window: int = 200,
    commission_bps: float = 10.0,
    starting_capital: float = 10_000.0,
    precleaned_data: pd.DataFrame = None,
) -> tuple:
    """
    Day 13: One-shot backtest runner that threads the full pipeline end-to-end
    for a given (ticker, date range, fast/slow window, cost) configuration.

    Returns (results_df, metrics_dict) so callers can either plot the result,
    dump it to a CSV, or aggregate many runs into a parameter sweep.

    Passing `precleaned_data` lets callers reuse a single download across many
    runs — a meaningful speedup when sweeping dozens of parameter combos.
    """
    if precleaned_data is None:
        raw = fetch_data(ticker, start_date, end_date)
        data = clean_data(raw, ticker)
    else:
        data = precleaned_data.copy()

    data = calculate_short_indicator(data, window=fast_window)
    data = calculate_long_indicator(data, window=slow_window)
    data = generate_signals(data)
    data = identify_trades(data)
    data = calculate_returns(data)
    data = apply_transaction_costs(data, commission_bps=commission_bps)
    data = build_equity_curve(data, starting_capital=starting_capital)
    metrics = compute_metrics(data)
    return data, metrics


def parameter_sweep(
    ticker: str,
    start_date: str,
    end_date: str,
    fast_windows: list,
    slow_windows: list,
    commission_bps: float = 10.0,
    metric: str = 'Sharpe Ratio',
) -> pd.DataFrame:
    """
    Day 13: Runs the backtest across every (fast, slow) pair in the grid and
    returns a matrix of the chosen metric (Sharpe by default).

    WHY THIS MATTERS — AND WHY IT IS ALSO DANGEROUS:
    A single backtest result (e.g. Sharpe 0.83 at 50/200) is one point in
    parameter space. It could be genuinely robust, or it could be a lucky
    local maximum that will break as soon as the future looks a little
    different. The sweep lets us see the whole terrain at once — if the
    neighbours of 50/200 also look good, we have some confidence the
    strategy is stable. If 50/200 is an isolated green dot in a sea of red,
    we have almost certainly overfit.

    IMPORTANT: This is still in-sample optimization. Picking the best cell
    from the heatmap and declaring victory is how amateurs get burned. A
    rigorous workflow would split the data, optimize on the first half, and
    then report the *unseen* second-half performance of the winning params.
    That walk-forward step is flagged in the roadmap.

    Only (fast < slow) combinations are evaluated. All noisy print output
    from the inner pipeline is suppressed so the sweep stays readable.
    """
    print(f"🔬 Running parameter sweep: "
          f"{len(fast_windows)} fast × {len(slow_windows)} slow windows "
          f"on {ticker} ({start_date} → {end_date})")

    # Download once, reuse across every run in the grid.
    raw = fetch_data(ticker, start_date, end_date)
    base = clean_data(raw, ticker)

    sweep = pd.DataFrame(index=fast_windows, columns=slow_windows, dtype=float)
    sweep.index.name = 'Fast window'
    sweep.columns.name = 'Slow window'

    total = sum(1 for f in fast_windows for s in slow_windows if f < s)
    done = 0
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue  # nonsensical: fast must be strictly shorter than slow
            # Silence the inner pipeline's chatty print() calls for one clean run.
            with contextlib.redirect_stdout(io.StringIO()):
                _, metrics = run_backtest(
                    ticker, start_date, end_date,
                    fast_window=fast, slow_window=slow,
                    commission_bps=commission_bps,
                    precleaned_data=base,
                )
            sweep.loc[fast, slow] = metrics[metric]
            done += 1
            print(f"   [{done:>2}/{total}] fast={fast:>3}  slow={slow:>3}  "
                  f"{metric}={metrics[metric]:>6.2f}")

    return sweep


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    ticker: str,
    metric_name: str = 'Sharpe Ratio',
    save_path: str = None,
) -> None:
    """
    Day 13: Renders the parameter sweep as a colored heatmap.

    Red = bad, green = good. Each cell is annotated with its numeric value
    so the reader can compare exact numbers, not just colors. Cells where
    fast >= slow are greyed out with NaN (there is no such thing as a
    crossover if the fast window is the same length or longer than the slow).
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    data = sweep_df.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', origin='lower')

    ax.set_xticks(range(len(sweep_df.columns)))
    ax.set_xticklabels(sweep_df.columns)
    ax.set_yticks(range(len(sweep_df.index)))
    ax.set_yticklabels(sweep_df.index)

    ax.set_xlabel('Slow SMA window (days)', fontsize=11)
    ax.set_ylabel('Fast SMA window (days)', fontsize=11)
    ax.set_title(
        f'{ticker} — {metric_name} across SMA window combinations\n'
        f'(higher = better; blank cells are invalid where fast ≥ slow)',
        fontsize=12, fontweight='bold'
    )

    # Annotate each cell with its numeric value for precise reading.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, color='black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"✅ Heatmap saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    TICKER = "SPY"
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"

    # ──────────────────────────────────────────────────────────────────
    # Part 1: Classic 50/200 crossover backtest (Days 1–12)
    # ──────────────────────────────────────────────────────────────────
    results, metrics = run_backtest(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        fast_window=50,
        slow_window=200,
        commission_bps=10.0,
        starting_capital=10_000.0,
    )

    print(f"\n--- Backtest Equity Curve ({TICKER}) ---")
    print(results[['Close', 'Signal', 'Strategy_Equity', 'BuyHold_Equity']].tail(10))
    print(f"\n--- Final Verdict ---")
    print(f"Strategy ending value:  ${results['Strategy_Equity'].iloc[-1]:>12,.2f}")
    print(f"Buy-and-Hold ending:    ${results['BuyHold_Equity'].iloc[-1]:>12,.2f}")

    print_metrics(metrics)
    plot_results(results, TICKER, save_path='backtest_result.png')

    # ──────────────────────────────────────────────────────────────────
    # Part 2 (Day 13): Parameter sweep across fast/slow SMA combinations
    # ──────────────────────────────────────────────────────────────────
    FAST_WINDOWS = [10, 20, 30, 40, 50, 60]
    SLOW_WINDOWS = [100, 150, 200, 250]

    sweep = parameter_sweep(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        fast_windows=FAST_WINDOWS,
        slow_windows=SLOW_WINDOWS,
        metric='Sharpe Ratio',
    )

    print("\n--- Parameter Sweep Results (Sharpe Ratio) ---")
    print(sweep.round(2))

    # Flag the single best combo — with the caveat that picking it on
    # in-sample data is exactly how overfitting happens.
    best = sweep.stack().idxmax()
    best_value = sweep.stack().max()
    print(f"\nHighest in-sample Sharpe: {best_value:.2f} at fast={best[0]}, slow={best[1]}")
    print("(Treat with caution — this is in-sample optimization.)")

    plot_sweep_heatmap(sweep, TICKER, metric_name='Sharpe Ratio',
                       save_path='parameter_sweep.png')
