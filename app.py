"""
QuantTradingBot — Interactive Streamlit Dashboard
===================================================
A comprehensive, interactive dashboard for comparing trading strategies,
visualizing backtests, and analyzing ML model performance.

Launch:  streamlit run app.py
Author:  Abhics8 (https://github.com/Abhics8)
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantTradingBot — ML-Powered Backtesting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #0f3460;
    }
    .stMetric label { color: #a8b2d1 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e6f1ff !important; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    .block-container { padding-top: 2rem; }
    h1 { color: #58a6ff !important; }
    h2, h3 { color: #79c0ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Imports (with error handling) ───────────────────────────────────
@st.cache_resource
def load_modules():
    """Import quantbot modules with graceful error handling."""
    modules = {}
    try:
        from quantbot.data.fetcher import fetch_ohlcv
        from quantbot.data.cleaner import clean_ohlcv
        from quantbot.strategies.sma_crossover import SMACrossover
        from quantbot.strategies.mean_reversion import BollingerMeanReversion
        from quantbot.strategies.momentum import RSIMomentum
        from quantbot.backtest.engine import BacktestEngine
        from quantbot.backtest.metrics import compute_metrics, compute_trade_metrics
        from quantbot.features.pipeline import build_feature_matrix, get_feature_columns
        from quantbot.portfolio.monte_carlo import MonteCarloSimulator

        modules["fetch_ohlcv"] = fetch_ohlcv
        modules["clean_ohlcv"] = clean_ohlcv
        modules["SMACrossover"] = SMACrossover
        modules["BollingerMeanReversion"] = BollingerMeanReversion
        modules["RSIMomentum"] = RSIMomentum
        modules["BacktestEngine"] = BacktestEngine
        modules["compute_metrics"] = compute_metrics
        modules["compute_trade_metrics"] = compute_trade_metrics
        modules["build_feature_matrix"] = build_feature_matrix
        modules["get_feature_columns"] = get_feature_columns
        modules["MonteCarloSimulator"] = MonteCarloSimulator
    except ImportError as e:
        st.error(f"Failed to import modules: {e}")
    return modules


@st.cache_data(ttl=3600)
def fetch_and_clean_data(ticker: str, start: str, end: str):
    """Fetch and cache market data."""
    from quantbot.data.fetcher import fetch_ohlcv
    from quantbot.data.cleaner import clean_ohlcv

    with contextlib.redirect_stdout(io.StringIO()):
        raw = fetch_ohlcv(ticker, start, end)
        clean = clean_ohlcv(raw, ticker)
    return clean


def run_strategy_backtest(strategy, data, commission_bps, starting_capital):
    """Run backtest with suppressed output."""
    from quantbot.backtest.engine import BacktestEngine

    engine = BacktestEngine(
        commission_bps=commission_bps,
        starting_capital=starting_capital,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        results, metrics = engine.run(strategy, data)
    return results, metrics


# ── Sidebar ─────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar controls."""
    st.sidebar.markdown("# 📈 QuantTradingBot")
    st.sidebar.markdown("**ML-Powered Backtesting Engine**")
    st.sidebar.markdown("---")

    # Ticker selection
    ticker = st.sidebar.text_input("🏷️ Ticker Symbol", value="SPY",
                                    help="Enter any stock/ETF ticker (e.g., SPY, AAPL, TSLA)")

    # Date range
    st.sidebar.markdown("### 📅 Date Range")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=pd.to_datetime("2018-01-01"))
    end_date = col2.date_input("End", value=pd.to_datetime("2024-01-01"))

    # Strategy selection
    st.sidebar.markdown("### 🎯 Strategies")
    strategies = {}
    strategies["sma"] = st.sidebar.checkbox("📊 SMA Crossover", value=True)
    strategies["bollinger"] = st.sidebar.checkbox("📉 Bollinger Mean Reversion", value=False)
    strategies["rsi"] = st.sidebar.checkbox("⚡ RSI Momentum", value=False)
    strategies["xgboost"] = st.sidebar.checkbox("🤖 XGBoost ML", value=False)
    strategies["lstm"] = st.sidebar.checkbox("🧠 LSTM Deep Learning", value=False)

    # Parameters
    st.sidebar.markdown("### ⚙️ Parameters")
    commission = st.sidebar.slider("Commission (bps)", 0, 50, 10)
    capital = st.sidebar.number_input("Starting Capital ($)", 1000, 1_000_000, 10_000, step=1000)

    # SMA-specific parameters
    if strategies["sma"]:
        st.sidebar.markdown("#### SMA Crossover")
        fast_window = st.sidebar.slider("Fast Window", 5, 100, 50)
        slow_window = st.sidebar.slider("Slow Window", 50, 400, 200)
    else:
        fast_window, slow_window = 50, 200

    # Monte Carlo
    st.sidebar.markdown("### 🎲 Advanced")
    run_monte_carlo = st.sidebar.checkbox("Run Monte Carlo Simulation", value=False)

    return {
        "ticker": ticker.upper(),
        "start": str(start_date),
        "end": str(end_date),
        "strategies": strategies,
        "commission": commission,
        "capital": capital,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "run_monte_carlo": run_monte_carlo,
    }


# ── Chart Functions ─────────────────────────────────────────────────

def plot_candlestick_with_signals(data: pd.DataFrame, results_dict: dict,
                                   ticker: str) -> go.Figure:
    """Create interactive candlestick chart with buy/sell signals."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{ticker} — Price & Signals", "Volume"],
    )

    # Candlestick
    if all(col in data.columns for col in ["Open", "High", "Low"]):
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="OHLC", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"],
            mode="lines", name="Close",
            line=dict(color="#58a6ff", width=1.5),
        ), row=1, col=1)

    # Add buy/sell markers for each strategy
    colors = ["#00ff88", "#ff6b6b", "#ffd93d", "#6bceff", "#c084fc"]
    for idx, (name, (results, _)) in enumerate(results_dict.items()):
        if "Position" in results.columns:
            buy_days = results[results["Position"] == 1.0]
            sell_days = results[results["Position"] == -1.0]
            color = colors[idx % len(colors)]

            fig.add_trace(go.Scatter(
                x=buy_days.index, y=buy_days["Close"],
                mode="markers", name=f"Buy ({name})",
                marker=dict(symbol="triangle-up", size=10, color=color),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=sell_days.index, y=sell_days["Close"],
                mode="markers", name=f"Sell ({name})",
                marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
            ), row=1, col=1)

    # Volume bars
    if "Volume" in data.columns:
        fig.add_trace(go.Bar(
            x=data.index, y=data["Volume"],
            name="Volume", marker_color="rgba(88, 166, 255, 0.3)",
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
    )
    return fig


def plot_equity_curves(results_dict: dict, starting_capital: float) -> go.Figure:
    """Plot equity curves for all strategies."""
    fig = go.Figure()

    colors = ["#00ff88", "#ff6b6b", "#ffd93d", "#6bceff", "#c084fc"]

    for idx, (name, (results, _)) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=results.index, y=results["Strategy_Equity"],
            mode="lines", name=name,
            line=dict(color=color, width=2),
        ))

    # Buy & Hold baseline (from first strategy)
    first_results = list(results_dict.values())[0][0]
    if "BuyHold_Equity" in first_results.columns:
        fig.add_trace(go.Scatter(
            x=first_results.index, y=first_results["BuyHold_Equity"],
            mode="lines", name="Buy & Hold",
            line=dict(color="#8b949e", width=1.5, dash="dash"),
        ))

    # Starting capital reference
    fig.add_hline(y=starting_capital, line_dash="dot",
                  line_color="#484f58", annotation_text="Starting Capital")

    fig.update_layout(
        title="📊 Equity Curve Comparison",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_drawdown(results_dict: dict) -> go.Figure:
    """Plot drawdown chart for all strategies."""
    fig = go.Figure()

    colors = ["#00ff88", "#ff6b6b", "#ffd93d", "#6bceff", "#c084fc"]

    for idx, (name, (results, _)) in enumerate(results_dict.items()):
        equity = results["Strategy_Equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100

        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=results.index, y=drawdown,
            mode="lines", name=name,
            fill="tozeroy",
            line=dict(color=color, width=1),
        ))

    fig.update_layout(
        title="📉 Drawdown Analysis",
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        template="plotly_dark",
        height=350,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_monte_carlo(simulations: np.ndarray, stats: dict,
                     starting_capital: float) -> go.Figure:
    """Plot Monte Carlo fan chart."""
    fig = go.Figure()

    n_days = simulations.shape[1]
    x = list(range(n_days))

    # Percentile bands
    p5 = np.percentile(simulations, 5, axis=0)
    p25 = np.percentile(simulations, 25, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p75 = np.percentile(simulations, 75, axis=0)
    p95 = np.percentile(simulations, 95, axis=0)

    # 5-95 band
    fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p5, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(88, 166, 255, 0.1)",
                             name="5th-95th percentile"))

    # 25-75 band
    fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(88, 166, 255, 0.25)",
                             name="25th-75th percentile"))

    # Median path
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines",
                             line=dict(color="#58a6ff", width=2),
                             name="Median"))

    # Starting capital line
    fig.add_hline(y=starting_capital, line_dash="dot",
                  line_color="#484f58", annotation_text="Starting Capital")

    fig.update_layout(
        title=f"🎲 Monte Carlo Simulation (1000 paths, 1 year)",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trading Days",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
    )
    return fig


def plot_feature_importance(importance: pd.Series) -> go.Figure:
    """Plot feature importance from ML model."""
    top_n = importance.head(15)

    fig = go.Figure(go.Bar(
        x=top_n.values,
        y=top_n.index,
        orientation="h",
        marker_color="#58a6ff",
    ))

    fig.update_layout(
        title="🔬 Feature Importance (XGBoost)",
        xaxis_title="Importance Score",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_monthly_returns_heatmap(results: pd.DataFrame) -> go.Figure:
    """Plot monthly returns heatmap."""
    returns = results["Strategy_Return"].dropna()
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    # Pivot to year x month
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = monthly_df.pivot_table(index="Year", columns="Month", values="Return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index.astype(str),
        colorscale="RdYlGn",
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont={"size": 11},
        colorbar_title="Return (%)",
    ))

    fig.update_layout(
        title="📅 Monthly Returns Heatmap",
        template="plotly_dark",
        height=300,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
    )
    return fig


# ── Main App ────────────────────────────────────────────────────────

def main():
    config = render_sidebar()

    # Header
    st.markdown("# 🚀 QuantTradingBot")
    st.markdown("**ML-Powered Quantitative Trading Research Platform**")
    st.markdown("---")

    # Run button
    if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
        st.session_state["run"] = True

    if not st.session_state.get("run", False):
        st.info("👈 Configure your backtest in the sidebar and click **Run Backtest** to begin.")

        # Show quick overview
        col1, col2, col3 = st.columns(3)
        col1.markdown("""
        ### 📊 Rule-Based Strategies
        - SMA Crossover (50/200)
        - Bollinger Mean Reversion
        - RSI Momentum
        """)
        col2.markdown("""
        ### 🤖 ML Strategies
        - XGBoost Signal Predictor
        - LSTM Deep Learning
        - Walk-Forward Validation
        """)
        col3.markdown("""
        ### 📈 Analysis Tools
        - Equity Curve Comparison
        - Drawdown Analysis
        - Monte Carlo Simulation
        """)
        return

    # ── Data Loading ────────────────────────────────────────────────
    with st.spinner(f"📡 Fetching {config['ticker']} data..."):
        try:
            data = fetch_and_clean_data(config["ticker"], config["start"], config["end"])
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    st.success(f"✅ Loaded {len(data)} trading days for {config['ticker']}")

    # ── Build & Run Strategies ──────────────────────────────────────
    from quantbot.strategies.sma_crossover import SMACrossover
    from quantbot.strategies.mean_reversion import BollingerMeanReversion
    from quantbot.strategies.momentum import RSIMomentum

    results_dict = {}  # {name: (results_df, metrics_dict)}
    active = config["strategies"]

    progress = st.progress(0, text="Running backtests...")
    total_strategies = sum(active.values())
    completed = 0

    if active.get("sma"):
        with st.spinner("📊 Running SMA Crossover..."):
            strategy = SMACrossover(config["fast_window"], config["slow_window"])
            res, met = run_strategy_backtest(strategy, data,
                                             config["commission"], config["capital"])
            results_dict[strategy.name] = (res, met)
            completed += 1
            progress.progress(completed / max(total_strategies, 1))

    if active.get("bollinger"):
        with st.spinner("📉 Running Bollinger Mean Reversion..."):
            strategy = BollingerMeanReversion(window=20, num_std=2.0)
            res, met = run_strategy_backtest(strategy, data,
                                             config["commission"], config["capital"])
            results_dict[strategy.name] = (res, met)
            completed += 1
            progress.progress(completed / max(total_strategies, 1))

    if active.get("rsi"):
        with st.spinner("⚡ Running RSI Momentum..."):
            strategy = RSIMomentum(rsi_period=14, oversold=30.0, overbought=70.0)
            res, met = run_strategy_backtest(strategy, data,
                                             config["commission"], config["capital"])
            results_dict[strategy.name] = (res, met)
            completed += 1
            progress.progress(completed / max(total_strategies, 1))

    if active.get("xgboost"):
        with st.spinner("🤖 Training XGBoost model (walk-forward validation)..."):
            try:
                from quantbot.models.xgboost_signal import XGBoostSignalModel, XGBoostStrategy
                from quantbot.features.pipeline import build_feature_matrix, get_feature_columns

                feature_cols = get_feature_columns()
                feature_df = build_feature_matrix(data.copy(), include_target=True)

                model = XGBoostSignalModel(n_estimators=200, max_depth=4, learning_rate=0.05)
                X = feature_df[feature_cols]
                y = feature_df["target"]
                model.fit(X, y)

                strategy = XGBoostStrategy(model=model, feature_columns=feature_cols)
                res, met = run_strategy_backtest(strategy, data,
                                                 config["commission"], config["capital"])
                results_dict[strategy.name] = (res, met)

                # Store feature importance for later display
                st.session_state["xgb_importance"] = model.feature_importance()
            except Exception as e:
                st.warning(f"XGBoost failed: {e}")

            completed += 1
            progress.progress(completed / max(total_strategies, 1))

    if active.get("lstm"):
        with st.spinner("🧠 Training LSTM model (this may take a moment)..."):
            try:
                from quantbot.models.lstm_signal import LSTMSignalModel, LSTMStrategy
                from quantbot.features.pipeline import build_feature_matrix, get_feature_columns

                feature_cols = get_feature_columns()
                feature_df = build_feature_matrix(data.copy(), include_target=True)

                model = LSTMSignalModel(seq_length=20, hidden_size=64, epochs=30, lr=0.001)
                X = feature_df[feature_cols]
                y = feature_df["target"]
                model.fit(X, y)

                strategy = LSTMStrategy(model=model, feature_columns=feature_cols)
                res, met = run_strategy_backtest(strategy, data,
                                                 config["commission"], config["capital"])
                results_dict[strategy.name] = (res, met)
            except Exception as e:
                st.warning(f"LSTM failed: {e}")

            completed += 1
            progress.progress(completed / max(total_strategies, 1))

    progress.empty()

    if not results_dict:
        st.error("❌ No strategies produced results.")
        return

    # ── Metrics Cards ───────────────────────────────────────────────
    st.markdown("## 📊 Performance Summary")

    for name, (res, met) in results_dict.items():
        st.markdown(f"### {name}")
        cols = st.columns(5)

        final_val = res["Strategy_Equity"].iloc[-1]
        total_ret = met.get("Total Return (%)", 0)
        sharpe = met.get("Sharpe Ratio", 0)
        max_dd = met.get("Max Drawdown (%)", 0)
        cagr = met.get("CAGR (%)", 0)

        cols[0].metric("💰 Final Value", f"${final_val:,.0f}",
                       delta=f"{total_ret:+.1f}%")
        cols[1].metric("📈 CAGR", f"{cagr:.1f}%")
        cols[2].metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}",
                       delta="Good" if sharpe > 1 else "Fair" if sharpe > 0.5 else "Poor")
        cols[3].metric("📉 Max Drawdown", f"{max_dd:.1f}%")
        cols[4].metric("📊 Win Rate",
                       f"{met.get('Win Rate (%)', 0):.1f}%" if 'Win Rate (%)' in met else "N/A")

    # ── Charts ──────────────────────────────────────────────────────
    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Signals", "💰 Equity Curves", "📉 Drawdown",
        "📅 Monthly Returns", "🔬 ML Insights",
    ])

    with tab1:
        fig = plot_candlestick_with_signals(data, results_dict, config["ticker"])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_equity_curves(results_dict, config["capital"])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_drawdown(results_dict)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        first_results = list(results_dict.values())[0][0]
        fig = plot_monthly_returns_heatmap(first_results)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if "xgb_importance" in st.session_state:
            fig = plot_feature_importance(st.session_state["xgb_importance"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable XGBoost strategy to see ML insights.")

    # ── Comparison Table ────────────────────────────────────────────
    if len(results_dict) > 1:
        st.markdown("## 📊 Strategy Comparison Table")

        comparison_data = []
        for name, (res, met) in results_dict.items():
            row = {"Strategy": name}
            row.update(met)
            row["Final Value ($)"] = res["Strategy_Equity"].iloc[-1]
            comparison_data.append(row)

        comp_df = pd.DataFrame(comparison_data).set_index("Strategy")

        # Format numeric columns
        styled = comp_df.style.format({
            col: "{:.2f}" for col in comp_df.columns if comp_df[col].dtype in ["float64", "float32"]
        }).background_gradient(
            subset=["Sharpe Ratio"],
            cmap="RdYlGn",
        )

        st.dataframe(styled, use_container_width=True)

    # ── Monte Carlo ─────────────────────────────────────────────────
    if config["run_monte_carlo"]:
        st.markdown("## 🎲 Monte Carlo Simulation")

        with st.spinner("Running 1000 simulations..."):
            from quantbot.portfolio.monte_carlo import MonteCarloSimulator

            # Use best strategy by Sharpe ratio
            best_name = max(results_dict, key=lambda k: results_dict[k][1].get("Sharpe Ratio", 0))
            best_returns = results_dict[best_name][0]["Strategy_Return"].dropna()

            simulator = MonteCarloSimulator(
                n_simulations=1000,
                n_days=252,
                starting_capital=config["capital"],
            )
            simulations = simulator.simulate(best_returns)
            stats = simulator.summary_statistics(simulations)

        # Display stats
        st.markdown(f"**Simulating {best_name} for 1 year forward:**")

        mc_cols = st.columns(4)
        mc_cols[0].metric("Median Final Value", f"${stats['median_final_value']:,.0f}")
        mc_cols[1].metric("5th Percentile", f"${stats['percentile_5']:,.0f}")
        mc_cols[2].metric("95th Percentile", f"${stats['percentile_95']:,.0f}")
        mc_cols[3].metric("Prob. of Profit", f"{stats['prob_profit']:.1%}")

        fig = plot_monte_carlo(simulations, stats, config["capital"])
        st.plotly_chart(fig, use_container_width=True)

    # ── Footer ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #8b949e; padding: 1rem;'>"
        "Built by <a href='https://github.com/Abhics8' style='color: #58a6ff;'>Abhics8</a> "
        "| QuantTradingBot | "
        "<a href='https://github.com/Abhics8/QuantTradingBot' style='color: #58a6ff;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
