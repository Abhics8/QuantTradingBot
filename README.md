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
9. **Stress-test the settings** — re-run the whole backtest across many fast/slow combinations and chart the results as a heatmap to see whether the strategy's success is robust or just a lucky pick of numbers.

Running `python ma_crossover.py` executes all of this in a few seconds.

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

Each run produces two charts:

**`backtest_result.png`** — a two-panel summary of the headline run:

* **Top:** the stock's price, overlaid with the 50-day and 200-day moving averages. Green triangles mark every day the program bought; red triangles mark every day it sold.
* **Bottom:** a side-by-side comparison of the strategy's portfolio value against a simple "buy and hold" baseline, so the difference is visible at a glance.

**`parameter_sweep.png`** — a heatmap showing how the strategy performs across many fast/slow window combinations. Green cells are good, red cells are bad, and each cell is labeled with its exact risk-adjusted score.

---

## How Sensitive Is the Strategy to Its Settings?

Why 50 and 200 days? Because that pair is the textbook standard. But if those particular numbers were the *only* ones that worked, that would be a red flag — it would mean the result was a fluke rather than a real pattern.

To check, the program automatically re-runs the entire backtest across a grid of 24 different combinations (fast windows from 10–60 days, slow windows from 100–250 days). Here is what the result looks like on the S&P 500 (2020–2022):

|  Fast \ Slow  | 100  | 150  | 200  | 250  |
| :-----------: | :--: | :--: | :--: | :--: |
| **10**        | 0.70 | 0.38 | 0.61 | 0.38 |
| **20**        | 0.71 | 0.72 | 0.63 | 0.36 |
| **30**        | 0.57 | 0.85 | 0.82 | 0.46 |
| **40**        | 0.85 | **0.89** | 0.70 | 0.74 |
| **50**        | 0.70 | 0.79 | 0.83 | 0.70 |
| **60**        | **0.92** | 0.79 | **0.97** | 0.46 |

> Each number is the risk-adjusted score for that combination of settings. The textbook 50/200 (0.83) is solid but not the absolute best on this data — 60/200 scores 0.97. More importantly, the entire region from 40–60 days fast and 100–200 days slow is consistently green. That clustering matters: a single high-scoring cell surrounded by red would suggest a lucky accident, but a whole green neighborhood suggests the underlying idea is genuine.

A quick word of caution: picking the single best cell from a chart like this and declaring victory is exactly how amateur backtesters fool themselves. The numbers above were optimized using the same data they were measured on. A truly rigorous test would freeze the chosen settings on this period and then re-measure them on a *different* period the model has never seen — which is the next planned upgrade.

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
├── backtest_result.png   # Headline equity-curve chart (regenerated each run)
├── parameter_sweep.png   # Heatmap across fast/slow window combinations
└── README.md             # This file
```

---

## Things It Doesn't Do (Yet)

Part of building something credible is being clear about what it isn't:

* **Only one stock at a time.** A real portfolio would hold many stocks, weighted thoughtfully.
* **Only goes long.** The program buys or sits in cash — it never bets against a falling market.
* **Simplified trading assumptions.** It assumes every trade can be executed at the previous day's closing price, which is close to, but not exactly, how a real broker would fill the order.
* **The parameter sweep is in-sample only.** The program tests many fast/slow combinations on the same period it then reports them on. A rigorous next step is *walk-forward validation* — pick the winner on the first half of the data, then verify it still works on the second half it has never seen.

---

## What's Next

* **Walk-forward validation** — split the data into "training" and "testing" halves, pick the winning settings on the first half only, and report the unseen second-half result honestly. This is the answer to the in-sample caveat above.
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
