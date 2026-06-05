#!/usr/bin/env python3
"""
QuantTradingBot — CLI Entry Point
==================================
Run backtests from the command line with any combination of strategies.

Examples:
    # Run SMA crossover on SPY
    python run_backtest.py --ticker SPY --strategies sma

    # Compare all strategies
    python run_backtest.py --ticker AAPL --strategies sma xgboost lstm bollinger rsi --start 2018-01-01

    # Run with custom parameters
    python run_backtest.py --ticker SPY --strategies sma --fast-window 20 --slow-window 100

Author:  Abhics8 (https://github.com/Abhics8)
License: MIT
"""

import argparse
import sys
import warnings
import contextlib
import io

import pandas as pd
import numpy as np

from quantbot.data.fetcher import fetch_ohlcv
from quantbot.data.cleaner import clean_ohlcv
from quantbot.strategies.sma_crossover import SMACrossover
from quantbot.strategies.mean_reversion import BollingerMeanReversion
from quantbot.strategies.momentum import RSIMomentum
from quantbot.backtest.engine import BacktestEngine
from quantbot.backtest.metrics import print_tearsheet
from quantbot.features.pipeline import build_feature_matrix, get_feature_columns


def build_ml_strategy(strategy_name: str, data: pd.DataFrame, feature_cols: list,
                      fast_mode: bool = False):
    """
    Build and train an ML-based strategy using walk-forward validation.

    Returns a Strategy object with a trained model, or None if dependencies
    are missing.
    """
    from quantbot.models.walk_forward import WalkForwardValidator
    from quantbot.features.pipeline import build_feature_matrix, get_feature_columns

    print(f"\n🧠 Training {strategy_name} model with walk-forward validation...")

    # Build feature matrix
    feature_df = build_feature_matrix(data.copy(), include_target=True)

    if strategy_name == "xgboost":
        from quantbot.models.xgboost_signal import XGBoostSignalModel, XGBoostStrategy

        model = XGBoostSignalModel(
            n_estimators=100 if fast_mode else 200,
            max_depth=4,
            learning_rate=0.05,
        )

        # Walk-forward validation
        validator = WalkForwardValidator(
            train_period_days=504,
            test_period_days=63,
            min_train_size=252,
        )
        wf_results = validator.validate(
            model_class=XGBoostSignalModel,
            model_kwargs={"n_estimators": 100 if fast_mode else 200,
                          "max_depth": 4, "learning_rate": 0.05},
            feature_df=feature_df,
            feature_cols=feature_cols,
        )

        print(f"   ✅ Walk-forward accuracy: {wf_results['aggregate_metrics']['accuracy']:.1%}")

        # Train final model on all data for signal generation
        X = feature_df[feature_cols]
        y = feature_df["target"]
        model.fit(X, y)

        return XGBoostStrategy(model=model, feature_columns=feature_cols)

    elif strategy_name == "lstm":
        try:
            from quantbot.models.lstm_signal import LSTMSignalModel, LSTMStrategy
        except ImportError:
            print("   ⚠️ PyTorch not installed. Skipping LSTM strategy.")
            return None

        model = LSTMSignalModel(
            seq_length=20,
            hidden_size=64,
            epochs=20 if fast_mode else 50,
            lr=0.001,
        )

        # Walk-forward validation
        validator = WalkForwardValidator(
            train_period_days=504,
            test_period_days=63,
            min_train_size=252,
        )
        wf_results = validator.validate(
            model_class=LSTMSignalModel,
            model_kwargs={"seq_length": 20, "hidden_size": 64,
                          "epochs": 20 if fast_mode else 50},
            feature_df=feature_df,
            feature_cols=feature_cols,
        )

        print(f"   ✅ Walk-forward accuracy: {wf_results['aggregate_metrics']['accuracy']:.1%}")

        # Train final model
        X = feature_df[feature_cols]
        y = feature_df["target"]
        model.fit(X, y)

        return LSTMStrategy(model=model, feature_columns=feature_cols)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="QuantTradingBot — ML-powered backtesting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --ticker SPY --strategies sma
  python run_backtest.py --ticker AAPL --strategies sma xgboost bollinger
  python run_backtest.py --ticker SPY --strategies sma --fast-window 20 --slow-window 100
        """,
    )

    parser.add_argument("--ticker", type=str, default="SPY",
                        help="Stock/ETF ticker symbol (default: SPY)")
    parser.add_argument("--start", type=str, default="2018-01-01",
                        help="Backtest start date (default: 2018-01-01)")
    parser.add_argument("--end", type=str, default="2024-01-01",
                        help="Backtest end date (default: 2024-01-01)")
    parser.add_argument("--strategies", nargs="+",
                        choices=["sma", "xgboost", "lstm", "bollinger", "rsi"],
                        default=["sma"],
                        help="Strategies to backtest (default: sma)")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Starting capital (default: 10000)")
    parser.add_argument("--commission", type=float, default=10.0,
                        help="Commission in basis points (default: 10)")
    parser.add_argument("--fast-window", type=int, default=50,
                        help="SMA fast window (default: 50)")
    parser.add_argument("--slow-window", type=int, default=200,
                        help="SMA slow window (default: 200)")
    parser.add_argument("--fast-mode", action="store_true",
                        help="Use faster ML training (fewer epochs/estimators)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save plots to PNG files instead of showing")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulation on best strategy")

    args = parser.parse_args()

    # ── Header ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  🚀 QuantTradingBot — ML-Powered Backtesting Engine")
    print("=" * 60)
    print(f"  Ticker:      {args.ticker}")
    print(f"  Period:      {args.start} → {args.end}")
    print(f"  Strategies:  {', '.join(args.strategies)}")
    print(f"  Capital:     ${args.capital:,.0f}")
    print(f"  Commission:  {args.commission} bps")
    print("=" * 60)

    # ── Fetch Data ──────────────────────────────────────────────────
    raw_data = fetch_ohlcv(args.ticker, args.start, args.end)
    data = clean_ohlcv(raw_data, args.ticker)

    # ── Build Feature Matrix (needed for ML strategies) ─────────────
    feature_cols = get_feature_columns()
    has_ml = any(s in args.strategies for s in ["xgboost", "lstm"])

    # ── Build Strategies ────────────────────────────────────────────
    strategies = []

    for strat_name in args.strategies:
        if strat_name == "sma":
            strategies.append(SMACrossover(
                fast_window=args.fast_window,
                slow_window=args.slow_window,
            ))
        elif strat_name == "bollinger":
            strategies.append(BollingerMeanReversion(window=20, num_std=2.0))
        elif strat_name == "rsi":
            strategies.append(RSIMomentum(rsi_period=14, oversold=30.0, overbought=70.0))
        elif strat_name in ("xgboost", "lstm"):
            ml_strat = build_ml_strategy(strat_name, data, feature_cols,
                                         fast_mode=args.fast_mode)
            if ml_strat is not None:
                strategies.append(ml_strat)

    if not strategies:
        print("❌ No valid strategies to run. Exiting.")
        sys.exit(1)

    # ── Run Backtests ───────────────────────────────────────────────
    engine = BacktestEngine(
        commission_bps=args.commission,
        starting_capital=args.capital,
    )

    if len(strategies) == 1:
        results, metrics = engine.run(strategies[0], data)
        print(f"\n📊 Results for: {strategies[0].name}")
        print_tearsheet(metrics)

        print(f"\n💰 Final Portfolio Value: ${results['Strategy_Equity'].iloc[-1]:,.2f}")
        print(f"📈 Buy & Hold Value:     ${results['BuyHold_Equity'].iloc[-1]:,.2f}")

    else:
        # Compare multiple strategies
        comparison = engine.compare(strategies, data)
        print("\n" + "=" * 70)
        print("  📊 STRATEGY COMPARISON")
        print("=" * 70)
        print(comparison.to_string())
        print("=" * 70)

    # ── Monte Carlo Simulation ──────────────────────────────────────
    if args.monte_carlo:
        from quantbot.portfolio.monte_carlo import MonteCarloSimulator

        print("\n🎲 Running Monte Carlo simulation...")
        simulator = MonteCarloSimulator(
            n_simulations=1000,
            n_days=252,
            starting_capital=args.capital,
        )

        # Use the last (or best) strategy's returns
        strat_returns = results["Strategy_Return"].dropna()
        simulations = simulator.simulate(strat_returns)
        stats = simulator.summary_statistics(simulations)

        print(f"\n   Monte Carlo Results (1000 simulations, 1 year forward):")
        print(f"   Median final value:  ${stats['median_final_value']:,.2f}")
        print(f"   5th percentile:      ${stats['percentile_5']:,.2f}")
        print(f"   95th percentile:     ${stats['percentile_95']:,.2f}")
        print(f"   Probability of profit: {stats['prob_profit']:.1%}")

    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
