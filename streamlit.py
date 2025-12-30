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