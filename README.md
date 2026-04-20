# QuantTradingBot

**A fully-vectorized Python backtesting engine for a 50 / 200-day Moving Average Crossover strategy on U.S. equities — engineered with production-grade discipline around look-ahead bias, transaction costs, and risk-adjusted performance measurement.**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-vectorized-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-visualization-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/yfinance-market%20data-8A2BE2" alt="yfinance">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Headline Results — SPY, 2020-01-01 → 2023-01-01

Starting capital **$10,000**. Transaction costs **10 bps per execution** (retail-realistic).

| Metric                  | Strategy    | Buy & Hold   |
| ----------------------- | ----------- | ------------ |
| Ending Equity           | **$12,731.85** | $12,348.60 |
| Total Return            | **+27.32%**   | +23.49%    |
| CAGR                    | **+8.40%**    | +7.29%     |
| Annualized Volatility   | **10.36%**    | 22.05%     |
| **Sharpe Ratio**        | **0.83**      | 0.35       |
| **Max Drawdown**        | **-12.87%**   | -33.72%    |

> The strategy extracts the bulk of the market's upside while side-stepping the 2022 drawdown — cutting peak-to-trough pain by more than half and more than doubling the risk-adjusted return (Sharpe). Raw return is comparable to buy-and-hold; the alpha is in the risk profile.

---

## Methodology

The engine is built as a composable, vectorized pipeline. Each stage is a pure function that takes a DataFrame and returns a DataFrame with new columns appended — making every intermediate step inspectable and unit-testable.

```
fetch_data  →  clean_data  →  calculate_short_indicator  →  calculate_long_indicator
           →  generate_signals  →  identify_trades  →  calculate_returns
           →  apply_transaction_costs  →  build_equity_curve
           →  compute_metrics  →  plot_results
```

### Signal Generation
* **Fast window:** 50-day Simple Moving Average
* **Slow window:** 200-day Simple Moving Average
* **Long entry (Golden Cross):** fast SMA crosses *above* slow SMA
* **Flat exit (Death Cross):** fast SMA crosses *below* slow SMA
* No leverage, no shorting, no intraday exposure — a single binary {0, 1} position vector.

### Risk & Return
* Daily returns computed as `pct_change()` on closing prices.
* Strategy P&L is `Signal.shift(1) * Market_Return` — the `shift(1)` is deliberate: signals are derived from the closing print, so the earliest executable fill is the *next* session's open.
* Equity compounded geometrically via `(1 + r).cumprod()`.

### Performance Metrics
Sharpe ratio is annualized with `√252`. Max drawdown is computed from the running cumulative-max of the strategy equity curve. All metrics are computed from post-cost returns.

---

## Engineering Principles

This project was built to demonstrate *how* a backtest is written, not just *what* it returns. Three disciplines drive the codebase:

### 1. Zero look-ahead bias
Both the signal-vs-return alignment (`shift(1)`) and the SMA rolling windows (no `min_periods=1`) are structured so the strategy cannot use any information it would not have had in real time. A NaN prefix on the first 199 days is retained as correct behavior — *not* patched away.

### 2. Realistic frictions
Transaction costs are modeled explicitly as a basis-point charge on every position change (`|ΔPosition|`). Default is 10 bps, which approximates retail commission plus typical bid-ask slippage on a liquid ETF. Infinite-liquidity, zero-cost backtests are rejected on principle — they inflate returns in a way that rarely survives live trading.

### 3. Vectorization over iteration
Every transformation is expressed as a column operation on the full price series rather than a Python `for` loop. The full SPY backtest runs end-to-end in well under a second, and the same codebase would scale to a basket of 500 tickers without structural change.

---

## Visual Output

Running `python ma_crossover.py` renders a two-panel summary chart (`backtest_result.png`):

* **Top panel** — close price overlaid with the 50/200-day SMAs, annotated with green ▲ markers on Golden Cross entries and red ▼ markers on Death Cross exits.
* **Bottom panel** — strategy equity curve (net of costs) plotted against the buy-and-hold benchmark so outperformance is visually immediate.

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/Abhics8/QuantTradingBot.git
cd QuantTradingBot

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backtest
python ma_crossover.py
```

The default run targets SPY from 2020-01-01 to 2023-01-01. Change the `TICKER`, `START_DATE`, and `END_DATE` constants in `__main__` to rerun on any other symbol or window.

---

## Project Structure

```
QuantTradingBot/
├── ma_crossover.py       # End-to-end vectorized backtest engine
├── requirements.txt      # yfinance, pandas, numpy, matplotlib
├── backtest_result.png   # Generated on each run
└── README.md
```

---

## Known Limitations

A deliberately honest list — each of these is a candidate for the next iteration of the engine:

* **Single-asset, long-only.** No portfolio construction, no shorting, no leverage.
* **Execution model is simplified.** The shift-by-one-day assumption treats the next open as perfectly fillable at the prior close — no gap risk, no partial fills.
* **Transaction costs are a flat bps.** Real costs are a function of order size, venue, and liquidity regime.
* **No parameter optimization / walk-forward validation.** The 50/200 windows are canonical but have not been cross-validated out-of-sample, which means the reported metrics have not been hardened against overfitting.
* **No benchmark for risk-free rate.** Sharpe is computed with `rf = 0`. Using a rolling T-Bill yield would be more rigorous.

---

## Roadmap

* Multi-asset portfolio backtesting with configurable weights
* Walk-forward parameter optimization (grid search over fast/slow windows)
* Additional signal families: mean-reversion, volatility breakout, pairs / cointegration
* HTML tear-sheet export (drawdown curves, rolling Sharpe, monthly return heatmap)
* Live paper-trading adapter via the Alpaca API

---

## Author

**AB0204** — [github.com/Abhics8](https://github.com/Abhics8)

Built as part of a structured, multi-week exploration of quantitative trading fundamentals: from raw price ingestion through to risk-adjusted performance reporting. Feedback, issues, and pull requests are welcome.

---

<sub>This project is for educational and research purposes only. Nothing in this repository constitutes investment advice or a solicitation to trade any financial instrument. Past performance — backtested or otherwise — is not indicative of future results.</sub>
