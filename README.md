# 📈 QuantTradingBot

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-green.svg)

A beginner-friendly Quantitative Trading algorithmic backtester. This engine simulates a classic **Moving Average Crossover** trading strategy using historical stock data, realistically accounting for transaction costs, to evaluate its performance against a baseline buy-and-hold approach.

*(Note: This project is currently being built step-by-step!)*

## 💡 How It Works

This algorithm relies on tracking two Simple Moving Averages (SMAs):
1. **The Fast Window (50-Day):** Reacts quickly to short-term price momentum.
2. **The Slow Window (200-Day):** Represents the overarching longer-term macroeconomic trend.

* **Buy Signal (Golden Cross):** Triggers when the Fast Window crosses *above* the Slow Window. 
* **Sell Signal (Death Cross):** Triggers when the Fast Window crosses *below* the Slow Window to protect capital.

## 🛠 Tech Stack
* **Python** (Core application logic)
* **Pandas & NumPy** (High-speed vectorized mathematical computations)
* **YFinance** (Ingestion of historical market timeline data)
* **Matplotlib** (Visualization of the equity curve)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhics8/QuantTradingBot.git
   cd QuantTradingBot
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```

3. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the backtest engine:**
   ```bash
   python ma_crossover.py
   ```

## 🧠 Core Academic Concepts Demonstrated
* Eradicating **Look-Ahead Bias** using statistical signal shifting.
* **Algorithmic Vectorization** to replace slow, iterative `for-loops` in Python.
* Applying simulated **Microstructure Friction (Transaction Costs)** to create a realistic model rather than an overly optimistic theoretical return.
