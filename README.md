# 📈 QuantTradingBot

QuantTradingBot is an ML-powered quantitative trading research platform and backtesting engine.

Initially built as a simple Moving Average Crossover tutorial, this project has been significantly upgraded to include machine learning signal prediction (XGBoost, LSTM), multi-strategy support, portfolio optimization, walk-forward validation, and an interactive Streamlit dashboard.

## 🚀 Features

*   **Machine Learning Signals:** Predict next-day returns using XGBoost classifiers and PyTorch LSTMs trained on technical indicators.
*   **Multiple Strategies:** Includes SMA Crossover, Bollinger Band Mean Reversion, RSI Momentum, and ML-powered strategies.
*   **Interactive Dashboard:** Streamlit app (`app.py`) for visualizing backtests, comparing strategies side-by-side, and analyzing equity curves and drawdowns interactively.
*   **Professional Metrics:** Computes Sharpe, Sortino, Calmar ratios, Win Rate, and Max Drawdown.
*   **Robust Validation:** Anchored walk-forward cross-validation prevents data leakage and overfitting.
*   **Monte Carlo Simulation:** Estimate future portfolio performance with thousands of bootstrapped simulations.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhics8/QuantTradingBot.git
    cd QuantTradingBot
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

### Streamlit Dashboard (Recommended)

The easiest way to interact with the backtester is via the UI:

```bash
streamlit run app.py
```

### CLI Runner

You can also run backtests from the command line:

```bash
# Run basic SMA strategy
python run_backtest.py --ticker SPY --strategies sma

# Compare multiple strategies including ML
python run_backtest.py --ticker AAPL --strategies sma xgboost bollinger --start 2018-01-01

# Run Monte Carlo simulation on the best strategy
python run_backtest.py --ticker SPY --strategies sma xgboost --monte-carlo
```

## 🏗️ Project Structure

*   `app.py`: Streamlit interactive dashboard.
*   `run_backtest.py`: Command-line backtest orchestrator.
*   `quantbot/`: Core Python package.
    *   `data/`: Data downloading and cleaning (Yahoo Finance).
    *   `features/`: Technical indicators (RSI, MACD, BB, ATR) and feature engineering.
    *   `models/`: ML models (XGBoost, LSTM) and walk-forward validation.
    *   `strategies/`: Trading strategies logic.
    *   `backtest/`: Backtest engine and performance metrics.
    *   `portfolio/`: Mean-variance optimization and Monte Carlo simulation.
*   `ma_crossover.py`: The original, single-file instructional version of the project.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
