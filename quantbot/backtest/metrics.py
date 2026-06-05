import pandas as pd
import numpy as np

def compute_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Computes professional backtest tear-sheet metrics.
    Includes standard metrics plus Calmar Ratio, Sortino Ratio.
    """
    print("📐 Computing performance tear-sheet metrics...")

    strat_returns = df['Strategy_Return'].dropna()
    strat_equity = df['Strategy_Equity'].dropna()

    if len(strat_returns) == 0 or strat_equity.empty:
        return {}

    # --- Total + Annualized Return ---
    total_return = strat_equity.iloc[-1] / strat_equity.iloc[0] - 1
    years = len(strat_returns) / 252.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # --- Volatility ---
    ann_vol = strat_returns.std() * np.sqrt(252)

    # --- Sharpe Ratio ---
    daily_rf = risk_free_rate / 252.0
    excess_returns = strat_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0

    # --- Max Drawdown ---
    running_peak = strat_equity.cummax()
    drawdown_series = (strat_equity - running_peak) / running_peak
    max_drawdown = drawdown_series.min()

    # --- Calmar Ratio ---
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # --- Sortino Ratio ---
    negative_returns = excess_returns[excess_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino = (excess_returns.mean() * 252) / downside_std if downside_std > 0 else 0.0

    metrics = {
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr * 100,
        'Annualized Volatility (%)': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown (%)': max_drawdown * 100,
    }
    
    # Merge trade metrics
    trade_metrics = compute_trade_metrics(df)
    metrics.update(trade_metrics)
    
    return metrics

def compute_trade_metrics(df: pd.DataFrame) -> dict:
    """
    Computes trade-specific metrics: win rate, profit factor, num trades.
    """
    if 'Position' not in df.columns:
        return {}
        
    trades = df[df['Position'] != 0.0].copy()
    if trades.empty:
         return {'Win Rate (%)': 0.0, 'Profit Factor': 0.0, 'Total Trades': 0}

    # Simplistic approximation: looking at returns on days we held the position
    # A more rigorous approach tracks exact entry/exit pairs.
    # For now, let's use the daily returns while in the market.
    
    # Filter days where strategy return is non-zero (implies market exposure)
    active_days = df[df['Signal'].shift(1) == 1.0]['Strategy_Return']
    
    winning_days = active_days[active_days > 0]
    losing_days = active_days[active_days < 0]
    
    gross_profit = winning_days.sum()
    gross_loss = abs(losing_days.sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    win_rate = (len(winning_days) / len(active_days)) * 100 if len(active_days) > 0 else 0.0
    
    # Number of trades is half the number of position changes (entry + exit)
    num_trades = int(len(trades) / 2)
    
    return {
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Total Trades': num_trades
    }

def print_tearsheet(metrics: dict) -> None:
    """Pretty-prints the metrics dictionary as a readable tear-sheet."""
    print("\n" + "=" * 45)
    print("      STRATEGY PERFORMANCE TEAR-SHEET")
    print("=" * 45)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:<30}{value:>10.2f}")
        else:
             print(f"  {name:<30}{value:>10}")
    print("=" * 45)
