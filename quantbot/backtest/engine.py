import pandas as pd
from quantbot.strategies.base import Strategy
from quantbot.backtest.metrics import compute_metrics

class BacktestEngine:
    """
    Core backtesting engine. Runs the full pipeline for a given strategy.
    """
    def __init__(self, commission_bps: float = 10.0, starting_capital: float = 10_000.0):
        self.commission_bps = commission_bps
        self.starting_capital = starting_capital
        
    def run(self, strategy: Strategy, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Run full backtest pipeline. 
        Returns (results_df, metrics_dict).
        """
        df = data.copy()
        
        # 1. Generate Signals
        df['Signal'] = strategy.generate_signals(df)
        
        # 2. Identify Trades
        df = self._identify_trades(df)
        
        # 3. Calculate Returns
        df = self._calculate_returns(df)
        
        # 4. Apply Transaction Costs
        df = self._apply_transaction_costs(df)
        
        # 5. Build Equity Curve
        df = self._build_equity_curve(df)
        
        # 6. Compute Metrics
        metrics = compute_metrics(df)
        
        return df, metrics
        
    def compare(self, strategies: list[Strategy], data: pd.DataFrame) -> pd.DataFrame:
        """
        Run multiple strategies and return comparison metrics table.
        """
        comparison_data = []
        for strategy in strategies:
            _, metrics = self.run(strategy, data)
            row = {"Strategy": strategy.name}
            row.update(metrics)
            comparison_data.append(row)
            
        return pd.DataFrame(comparison_data).set_index("Strategy")
        
    def _identify_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔍 Identifying specific trade execution moments...")
        # Calculate the day-to-day discrete difference in the Signal column
        df['Position'] = df['Signal'].diff().fillna(0.0)
        return df
        
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        print("💰 Calculating daily market and strategy returns...")
        # Daily percentage change of the close price = buy-and-hold return
        df['Market_Return'] = df['Close'].pct_change()

        # Strategy return = yesterday's position decision * today's realized return
        # CRITICAL: We shift the Signal by 1 day to avoid look-ahead bias!
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Market_Return']
        return df
        
    def _apply_transaction_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        cost_rate = self.commission_bps / 10_000.0
        print(f"💸 Applying transaction costs at {self.commission_bps:.1f} bps per trade...")

        # Position is the diff of Signal: +1 on buys, -1 on sells, 0 when holding
        trade_cost = df['Position'].abs() * cost_rate
        df['Strategy_Return'] = df['Strategy_Return'] - trade_cost.fillna(0)
        return df
        
    def _build_equity_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"📊 Compounding equity curve from ${self.starting_capital:,.0f} starting capital...")

        # fillna(0) is safe here
        strategy_growth = (1 + df['Strategy_Return'].fillna(0)).cumprod()
        benchmark_growth = (1 + df['Market_Return'].fillna(0)).cumprod()

        df['Strategy_Equity'] = self.starting_capital * strategy_growth
        df['BuyHold_Equity'] = self.starting_capital * benchmark_growth
        return df
