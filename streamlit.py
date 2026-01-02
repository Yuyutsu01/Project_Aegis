import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Project AEGIS — Hard Consensus Trading",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Project AEGIS — Hard Consensus Trading System")
st.caption(
    "A research-grade hybrid trading system combining XGBoost, PPO, and Sentiment with strict hard-consensus execution."
)

# =========================
# LOAD DATA
# =========================
DATA_PATH = "artifacts/backtests/hard_consensus_results.csv"
df = pd.read_csv(DATA_PATH)

# =========================
# METRICS (READ-ONLY)
# =========================
equity = df["equity"].values
returns = np.diff(equity) / equity[:-1]

cum_return = equity[-1] - 1
years = len(equity) / 252
cagr = (equity[-1]) ** (1 / years) - 1 if years > 0 else 0
max_dd = df["drawdown"].min()

sharpe = (
    np.sqrt(252) * returns.mean() / returns.std()
    if returns.std() > 0 else 0
)

calmar = cagr / abs(max_dd) if max_dd < 0 else 0
exposure = (df["position"] != 0).mean()
trade_count = (df["position"].diff().abs() > 0).sum()

# =========================
# KPI ROW (ANIMATED)
# =========================
st.subheader("📊 Strategy Performance Overview")

k1, k2, k3, k4, k5 = st.columns(5)

with st.spinner("Loading metrics..."):
    time.sleep(0.4)

k1.metric(
    "Cumulative Return",
    f"{cum_return*100:.2f}",
    help="Total return over the full backtest horizon."
)

k2.metric(
    "Annualised Return",
    f"{cagr*100:.2f}%",
    help="Compound annual growth rate (CAGR)."
)

k3.metric(
    "Sharpe Ratio",
    f"{sharpe:.2f}",
    help="Risk-adjusted return using total volatility."
)

k4.metric(
    "Calmar Ratio",
    f"{calmar:.2f}",
    help="Annualised return divided by maximum drawdown."
)

k5.metric(
    "Exposure",
    f"{exposure*100:.2f}%",
    help="Fraction of time the strategy is invested."
)

# =========================
# EQUITY CURVE (INTERACTIVE)
# =========================
st.subheader("Equity Curve")

fig_equity = go.Figure()
fig_equity.add_trace(
    go.Scatter(
        y=df["equity"],
        mode="lines",
        name="Equity",
        line=dict(width=2)
    )
)
fig_equity.update_layout(
    height=400,
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Equity",
    template="plotly_dark"
)

st.plotly_chart(fig_equity, use_container_width=True)

# =========================
# DRAWDOWN CURVE
# =========================
st.subheader("Drawdown")

fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(
        y=df["drawdown"],
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="crimson")
    )
)

fig_dd.update_layout(
    height=250,
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Drawdown",
    template="plotly_dark"
)

st.plotly_chart(fig_dd, use_container_width=True)

# =========================
# POSITION & SIGNALS
# =========================
st.subheader("🧠 Model Signals & Executed Position")

fig_sig = go.Figure()

fig_sig.add_trace(go.Scatter(y=df["position"], name="Position", line=dict(width=3)))
fig_sig.add_trace(go.Scatter(y=df["xgb_signal"], name="XGBoost Signal", opacity=0.5))
fig_sig.add_trace(go.Scatter(y=df["ppo_signal"], name="PPO Signal", opacity=0.5))
fig_sig.add_trace(go.Scatter(y=df["sentiment_signal"], name="Sentiment Signal", opacity=0.5))

fig_sig.update_layout(
    height=350,
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Signal / Position",
    template="plotly_dark"
)

st.plotly_chart(fig_sig, use_container_width=True)

# =========================
# TRADE STATS
# =========================
st.subheader("Trade Activity")

c1, c2 = st.columns(2)

c1.metric(
    "Trades Taken",
    trade_count,
    help="Number of position changes (entries + exits)."
)

c2.metric(
    "Exposure Ratio",
    f"{exposure*100:.2f}%",
    help="Percentage of time the strategy holds a non-zero position."
)

# =========================
# RAW DATA (OPTIONAL)
# =========================
with st.expander("View Raw Backtest Data"):
    st.dataframe(df, use_container_width=True)
