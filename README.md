# QuantTradingBot

**A Python program that simulates a classic stock-trading strategy on real historical market data — and measures, honestly, whether it would have made money after fees.**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-data%20analysis-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-math-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-charts-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/yfinance-market%20data-8A2BE2" alt="yfinance">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## The Idea in One Paragraph

The strategy is simple and famous: follow two moving averages of the stock's price — a fast one (50 days) that reacts to recent moves, and a slow one (200 days) that shows the long-term trend. When the fast line rises above the slow line, it's a sign momentum is turning up, so the program buys. When it falls back below, the program sells and sits in cash. The aim is not to beat the market's raw return, but to sidestep the worst crashes while still capturing most of the upside.

---

## How It Performed on the S&P 500 (2020–2022)

Starting with **$10,000** and paying a small, realistic trading fee each time it bought or sold:

| What we measured            | The strategy   | Just holding the market |
| --------------------------- | -------------- | ----------------------- |
| Ending balance              | **$12,732**    | $12,349                 |
| Total growth                | **+27%**       | +23%                    |
| Growth per year (avg.)      | **+8.4%**      | +7.3%                   |
| Worst drop from a high      | **-13%**       | -34%                    |
| Risk-adjusted score¹        | **0.83**       | 0.35                    |

> The headline result isn't the extra few percent of profit. It's the **worst drop cut from -34% down to -13%** — a much smoother ride for a comparable return. A dollar of profit earned while avoiding a stomach-churning drawdown is, to most investors, worth more than a dollar earned through one.

<sub>¹ A standard measure of how much return you're getting for the risk you're taking. Higher is better; above 1.0 is generally considered strong.</sub>

---

## How the Program Works

The code is organized as a straight pipeline — each step is one small, readable function:

1. **Download** three years of daily prices for the chosen stock or index.
2. **Clean** the data (fill in missing days, standardize the format).
3. **Calculate** the 50-day and 200-day moving averages.
4. **Generate signals** — mark each day as either "in the market" or "in cash."
5. **Simulate trading** — track what would have happened day by day.
6. **Subtract fees** so the result reflects what an ordinary investor would actually keep.
7. **Measure performance** — growth rate, volatility, worst loss, risk-adjusted return.
8. **Plot** the result as a two-panel chart.

Running `python ma_crossover.py` executes all of this in under a second.

---

## What Sets This Project Apart

Plenty of tutorials walk through a simple backtest. This one goes further on three fronts that matter if you want the numbers to be trustworthy:

### 1. No cheating with hindsight
A surprisingly common mistake in amateur backtests is letting the simulated trader "see" today's price before deciding whether to buy it. This project is carefully structured so the trader only ever acts on information that would have been available in real time — the same rule a real broker would enforce.

### 2. Honest about fees
Every time a real investor buys or sells, they pay a small cost (broker commission plus the gap between bid and ask prices). The simulation applies that same cost on every trade, so the final number reflects what would actually land in your pocket — not an idealized figure that ignores friction.

### 3. Fast and clean code
The math runs on entire price histories at once rather than day-by-day loops, which is why it's quick and why the same codebase could scale to hundreds of stocks with no structural changes.

---

## What the Output Looks Like

Each run produces a chart (`backtest_result.png`) with two panels:

* **Top:** the stock's price, overlaid with the 50-day and 200-day moving averages. Green triangles mark every day the program bought; red triangles mark every day it sold.
* **Bottom:** a side-by-side comparison of the strategy's portfolio value against a simple "buy and hold" baseline, so the difference is visible at a glance.

---

## Running It Yourself

```bash
# 1. Download the code
git clone https://github.com/Abhics8/QuantTradingBot.git
cd QuantTradingBot

# 2. Set up an isolated Python environment
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# 3. Install the libraries it depends on
pip install -r requirements.txt

# 4. Run the backtest
python ma_crossover.py
```

The default run is the S&P 500 index (ticker: `SPY`) from 2020 through 2022. To test it on any other stock or date range, edit the `TICKER`, `START_DATE`, and `END_DATE` values at the bottom of `ma_crossover.py`.

---

## Project Files

```
QuantTradingBot/
├── ma_crossover.py       # The entire backtest program
├── requirements.txt      # List of Python libraries needed
├── backtest_result.png   # Chart produced each time you run it
└── README.md             # This file
```

---

## Things It Doesn't Do (Yet)

Part of building something credible is being clear about what it isn't:

* **Only one stock at a time.** A real portfolio would hold many stocks, weighted thoughtfully.
* **Only goes long.** The program buys or sits in cash — it never bets against a falling market.
* **Simplified trading assumptions.** It assumes every trade can be executed at the previous day's closing price, which is close to, but not exactly, how a real broker would fill the order.
* **No guarantee the 50/200 numbers are optimal.** These are the textbook defaults. A more rigorous version would test many combinations on historical data and verify the winner holds up on a *different* period it hasn't seen before.

---

## What's Next

* Testing the strategy across multiple stocks as a portfolio
* Trying other well-known strategies (mean-reversion, volatility breakouts)
* Producing a polished one-page performance report
* Connecting to a free paper-trading service to run it live without real money

---

## Author

**AB0204** — [github.com/Abhics8](https://github.com/Abhics8)

Built as part of a structured, multi-week project to learn quantitative trading from the ground up — starting with raw price data and ending with a real performance report. Feedback, questions, and contributions are welcome.

---

<sub>This project is for learning and research only. It is not investment advice, and the fact that a strategy worked in the past is not a promise it will work in the future.</sub>
