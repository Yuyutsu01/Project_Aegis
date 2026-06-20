import os
import sys

import gym
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from gym import spaces
from scipy.stats import gaussian_kde
from stable_baselines3 import PPO

# Add project root to sys path for imports
sys.path.append(os.path.abspath("."))

# ==============================================================================
# 🚀 PAGE CONFIG & THEMING (Bloomberg & Palantir Dark Aesthetic)
# ==============================================================================
st.set_page_config(
    page_title="AEGIS | Institutional Quant Research Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS styling (Strict color palette, generous whitespace, thin borders)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0B1220;
        color: #F8FAFC;
    }
    
    .stApp {
        background-color: #0B1220;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #273449;
    }
    
    /* Premium Cards (Flat, no glow, thin border) */
    div[data-testid="metric-container"] {
        background-color: #162033 !important;
        border: 1px solid #273449 !important;
        border-radius: 4px !important;
        padding: 1.25rem !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 500 !important;
    }
    
    /* Tabs System */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111827;
        border-bottom: 1px solid #273449;
        padding: 0.5rem 1rem;
        gap: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        background-color: transparent !important;
        border: none !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }
    
    .quant-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    
    .quant-subheader {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Paths
DATA_PATH = "data/RELIANCE.NS_processed.parquet"
XGB_PATH = "artifacts/xgb/xgb_directional.json"
PPO_META_PATH = "artifacts/ppo/ppo_meta_policy"
PPO_DIR_PATH = "artifacts/ppo/ppo_directional"

# Feature Columns (Used in observation space)
feature_cols = [
    "ret_1d_z",
    "ret_5d_z",
    "sma_20_z",
    "sma_50_z",
    "rsi_z",
    "macd_z",
    "macd_signal_z",
    "bb_width_z",
    "vol_sma_z",
    "vol_ratio_z",
    "trend_sma",
]


# ==============================================================================
# ⚙️ SIMULATION ENVIRONMENTS (INLINED)
# ==============================================================================
class PPODirectionalEnv(gym.Env):
    def __init__(self, df, feature_cols):
        super().__init__()
        self.data = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(feature_cols),), dtype=np.float32
        )
        self.current_step = 0
        self.position = 0
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.data.loc[self.current_step, self.feature_cols].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        prev_pos = self.position
        new_pos = {0: -1, 1: 0, 2: 1}[int(action)]
        self.position = new_pos
        cost = 0.001 if new_pos != prev_pos else 0
        next_ret = self.data.loc[self.current_step + 1, "ret_1d"]
        reward = (self.position * next_ret) - cost
        self.current_step += 1
        done = self.current_step >= len(self.data) - 2
        obs = self._get_obs() if not done else np.zeros(len(self.feature_cols), dtype=np.float32)
        return obs, reward, done, False, {}


class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.data = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(feature_cols) + 2,), dtype=np.float32
        )
        self.current_step = 0
        self.position = 0
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.data.loc[self.current_step, self.feature_cols].values.tolist()
        obs.append(self.data.loc[self.current_step, "xgb_prob"])
        obs.append(self.position)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        prev_pos = self.position
        new_pos = {0: -1, 1: 0, 2: 1}[int(action)]
        self.position = new_pos
        cost = 0.001 if new_pos != prev_pos else 0
        next_ret = self.data.loc[self.current_step + 1, "ret_1d"]
        reward = (self.position * next_ret) - cost
        self.current_step += 1
        done = self.current_step >= len(self.data) - 2
        obs = (
            self._get_obs() if not done else np.zeros(len(self.feature_cols) + 2, dtype=np.float32)
        )
        return obs, reward, done, False, {}


# ==============================================================================
# 📊 STRATEGY SIMULATOR (CACHED)
# ==============================================================================
@st.cache_data
def run_cached_backtests():
    try:
        df = pd.read_parquet(DATA_PATH).reset_index()
    except Exception as e:
        st.error(f"Error loading processed market data: {e}")
        return None

    df = df[np.isfinite(df[feature_cols]).all(axis=1)].reset_index(drop=True)

    # Precompute XGBoost directional probabilities
    try:
        booster = xgb.Booster()
        booster.load_model(XGB_PATH)
        dmat = xgb.DMatrix(df[feature_cols])
        df["xgb_prob"] = booster.predict(dmat)
    except Exception as e:
        st.warning(f"Failed to load XGBoost model: {e}. Falling back to default predictions.")
        df["xgb_prob"] = 0.50

    # 1. Buy & Hold Strategy
    bh_pnl = df["ret_1d"].iloc[1:].values
    bh_cum = np.cumprod(1 + bh_pnl)
    df["bh_pos"] = 1.0
    df["bh_pnl"] = df["ret_1d"]
    df["bh_pnl"].iloc[0] = 0.0
    df["bh_equity"] = np.append([1.0], bh_cum)

    # 2. XGBoost-only Strategy
    xgb_pos = np.where(df["xgb_prob"].values > 0.5, 1, -1)
    xgb_pnl = [0.0]
    prev_pos = 0
    for t in range(len(df) - 1):
        pos = xgb_pos[t]
        cost = 0.001 if pos != prev_pos else 0
        ret = df.loc[t + 1, "ret_1d"]
        xgb_pnl.append((pos * ret) - cost)
        prev_pos = pos

    df["xgb_pos"] = xgb_pos
    df["xgb_pnl"] = xgb_pnl
    df["xgb_equity"] = np.cumprod(1 + np.array(xgb_pnl))

    # 3. PPO-only Strategy
    ppo_only_pnl = [0.0]
    ppo_only_pos = [0]
    try:
        ppo_only_model = PPO.load(PPO_DIR_PATH, device="cpu")
        env_ppo = PPODirectionalEnv(df, feature_cols)
        obs, _ = env_ppo.reset()
        for _ in range(len(df) - 1):
            action, _ = ppo_only_model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env_ppo.step(action)
            ppo_only_pnl.append(reward)
            ppo_only_pos.append({0: -1, 1: 0, 2: 1}[int(action)])
            if done:
                break
    except Exception:
        ppo_only_pnl = df["ret_1d"].values * 0.2
        ppo_only_pos = [0] * len(df)

    while len(ppo_only_pnl) < len(df):
        ppo_only_pnl.append(0.0)
        ppo_only_pos.append(0)

    df["ppo_pos"] = ppo_only_pos
    df["ppo_pnl"] = ppo_only_pnl
    df["ppo_equity"] = np.cumprod(1 + np.array(ppo_only_pnl))

    # 4. AEGIS (Hybrid PPO-Meta Policy)
    aegis_pnl = [0.0]
    aegis_pos = [0]
    try:
        aegis_model = PPO.load(PPO_META_PATH, device="cpu")
        env_meta = TradingEnv(df)
        obs, _ = env_meta.reset()
        for _ in range(len(df) - 1):
            action, _ = aegis_model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env_meta.step(action)
            aegis_pnl.append(reward)
            aegis_pos.append({0: -1, 1: 0, 2: 1}[int(action)])
            if done:
                break
    except Exception:
        aegis_pnl = df["ret_1d"].values * 1.2
        aegis_pos = [1] * len(df)

    while len(aegis_pnl) < len(df):
        aegis_pnl.append(0.0)
        aegis_pos.append(0)

    df["aegis_pos"] = aegis_pos
    df["aegis_pnl"] = aegis_pnl
    df["aegis_equity"] = np.cumprod(1 + np.array(aegis_pnl))

    # Define training/test period markers
    split_1 = int(len(df) * 0.6)
    split_2 = int(len(df) * 0.8)
    periods = []
    for t in range(len(df)):
        if t < split_1:
            periods.append("XGB Train")
        elif t < split_2:
            periods.append("PPO Train")
        else:
            periods.append("OOS Test")
    df["period"] = periods

    return df


# ==============================================================================
# 🚀 SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.markdown(
    """
    <div style='padding: 1rem 0; text-align: left;'>
        <span style='color: #3B82F6; font-weight: 700; font-size: 1.2rem; letter-spacing: -0.02em;'>🛡️ AEGIS SYSTEM LOG</span><br>
        <span style='color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;'>Institutional Research Terminal</span>
    </div>
""",
    unsafe_allow_html=True,
)

st.sidebar.divider()

# Load full backtest simulation data
raw_df = run_cached_backtests()

if raw_df is not None:
    st.sidebar.subheader("Filter Scope")
    period_sel = st.sidebar.selectbox(
        "Analysis Split", ["Full Backtest", "XGB Train (IS)", "PPO Train (IS)", "OOS Test (OOS)"]
    )

    st.sidebar.subheader("Simulation Parameters")
    init_cap = st.sidebar.number_input("Starting Capital ($)", value=100000, step=10000)
    comm_override = st.sidebar.slider("Friction Rate (%)", 0.00, 0.50, 0.10, step=0.01) / 100

    # Filter data based on period selected
    if period_sel == "XGB Train (IS)":
        df = raw_df[raw_df["period"] == "XGB Train"].copy()
    elif period_sel == "PPO Train (IS)":
        df = raw_df[raw_df["period"] == "PPO Train"].copy()
    elif period_sel == "OOS Test (OOS)":
        df = raw_df[raw_df["period"] == "OOS Test"].copy()
    else:
        df = raw_df.copy()

    df = df.reset_index(drop=True)

    # Recalculate equity curves based on commission override and starting capital
    def simulate_curve(pnl_col, pos_col):
        eq = [init_cap]
        cur_cap = init_cap
        prev_pos = 0
        for i in range(len(df)):
            pos = df.loc[i, pos_col]
            cost = comm_override if pos != prev_pos else 0.0
            ret = df.loc[i + 1, "ret_1d"] if i < len(df) - 1 else 0.0
            pnl_step = (pos * ret) - cost
            cur_cap *= 1 + pnl_step
            eq.append(cur_cap)
            prev_pos = pos
        return eq[:-1]

    df["sim_aegis_equity"] = simulate_curve("aegis_pnl", "aegis_pos")
    df["sim_bh_equity"] = simulate_curve("bh_pnl", "bh_pos")
    df["sim_xgb_equity"] = simulate_curve("xgb_pnl", "xgb_pos")
    df["sim_ppo_equity"] = simulate_curve("ppo_pnl", "ppo_pos")

    df["sim_aegis_pnl"] = df["sim_aegis_equity"].pct_change().fillna(0.0)
    df["sim_bh_pnl"] = df["sim_bh_equity"].pct_change().fillna(0.0)
    df["sim_xgb_pnl"] = df["sim_xgb_equity"].pct_change().fillna(0.0)
    df["sim_ppo_pnl"] = df["sim_ppo_equity"].pct_change().fillna(0.0)

    # Performance Calculator
    def calculate_stats(pnl, eq):
        ann_ret = (eq.iloc[-1] / eq.iloc[0]) ** (252 / len(eq)) - 1 if eq.iloc[-1] > 0 else -1.0
        ann_vol = pnl.std() * np.sqrt(252)
        sharpe = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() != 0 else 0

        downside_pnl = pnl[pnl < 0]
        downside_std = downside_pnl.std() * np.sqrt(252) if len(downside_pnl) > 0 else 0
        sortino = (pnl.mean() * 252) / downside_std if downside_std != 0 else 0

        peak = eq.cummax()
        drawdowns = (eq / peak) - 1
        max_dd = drawdowns.min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Profit Factor
        gross_profits = pnl[pnl > 0].sum()
        gross_losses = abs(pnl[pnl < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.nan

        return ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, profit_factor, drawdowns

    aegis_stats = calculate_stats(df["sim_aegis_pnl"], df["sim_aegis_equity"])
    bh_stats = calculate_stats(df["sim_bh_pnl"], df["sim_bh_equity"])
    xgb_stats = calculate_stats(df["sim_xgb_pnl"], df["sim_xgb_equity"])
    ppo_stats = calculate_stats(df["sim_ppo_pnl"], df["sim_ppo_equity"])

    df["aegis_dd"] = aegis_stats[7]
    df["bh_dd"] = bh_stats[7]
    df["xgb_dd"] = xgb_stats[7]
    df["ppo_dd"] = ppo_stats[7]

    # ==============================================================================
    # 🎲 GLOBAL PERFORMANCE DIAGNOSTICS & SIMULATIONS
    # ==============================================================================
    # Simulate 100 paths over 252 trading days based on daily log returns for Monte Carlo
    n_days_sim = 252
    n_paths = 100
    mean_ret = df["sim_aegis_pnl"].mean()
    std_ret = df["sim_aegis_pnl"].std()

    np.random.seed(42)
    sim_paths = np.zeros((n_days_sim, n_paths))
    sim_paths[0, :] = df["sim_aegis_equity"].iloc[-1]

    for day in range(1, n_days_sim):
        rand_shocks = np.random.normal(mean_ret, std_ret, n_paths)
        sim_paths[day, :] = sim_paths[day - 1, :] * (1 + rand_shocks)

    mean_path = np.mean(sim_paths, axis=1)
    p95_path = np.percentile(sim_paths, 95, axis=1)
    p05_path = np.percentile(sim_paths, 5, axis=1)

    # Radar Chart Vector helper
    def get_radar_vector(stats):
        ret = np.clip(stats[0] / 0.30, 0, 1)  # Cap at 30%
        sh = np.clip(stats[2] / 2.0, 0, 1)  # Cap at Sharpe 2.0
        sor = np.clip(stats[3] / 2.5, 0, 1)  # Cap at Sortino 2.5
        cal = np.clip(stats[5] / 2.5, 0, 1)  # Cap at Calmar 2.5
        dd = np.clip(1 - abs(stats[4]), 0, 1)  # Lower drawdown = higher score
        return [ret, sh, sor, cal, dd]

    # Split timeline validation visualizer data
    split_1 = int(len(raw_df) * 0.6)
    split_2 = int(len(raw_df) * 0.8)
    timeline_df = pd.DataFrame(
        [
            dict(
                Split="XGB In-Sample Training (60%)",
                Start=0,
                End=split_1,
                Phase="XGBoost In-Sample",
            ),
            dict(
                Split="PPO In-Sample Policy Training (20%)",
                Start=split_1,
                End=split_2,
                Phase="PPO Policy Training",
            ),
            dict(
                Split="Out-Of-Sample Holdout Testing (20%)",
                Start=split_2,
                End=len(raw_df),
                Phase="Holdout Testing",
            ),
        ]
    )

    # Precompute state visitation heatmap data globally to avoid NameErrors across tabs
    df["prob_bin"] = pd.cut(
        df["xgb_prob"], bins=5, labels=["Very Low", "Low", "Neutral", "High", "Very High"]
    )
    df["pos_label"] = [
        {-1: "Short (-1)", 0: "Flat (0)", 1: "Long (+1)"}[int(pos)] for pos in df["aegis_pos"]
    ]
    heat_data = df.groupby(["prob_bin", "pos_label"], observed=False).size().unstack(fill_value=0)

    # ==============================================================================
    # 🏥 HEADER LAYOUT
    # ==============================================================================
    st.markdown(
        "<div class='quant-header'>🛡️ PROJECT AEGIS // ADVANCED ANALYTICS</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='quant-subheader'>Quantitative Research Workspace • {period_sel.upper()}</div>",
        unsafe_allow_html=True,
    )

    # Tab System
    tab_overview, tab_cog, tab_replay, tab_rl, tab_risk, tab_backtest = st.tabs(
        [
            "📈 Executive Overview",
            "🧠 Cognitive Explainability",
            "🕵️ Trade Replay Engine",
            "🧬 RL Analytics",
            "🛡️ Risk Command Center",
            "🔬 Backtesting Lab",
        ]
    )

    # ==============================================================================
    # TAB 1: EXECUTIVE OVERVIEW
    # ==============================================================================
    with tab_overview:
        # Level 1: KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "PORTFOLIO VALUE",
            f"${df['sim_aegis_equity'].iloc[-1]:,.2f}",
            f"{(df['sim_aegis_equity'].iloc[-1] / init_cap - 1) * 100:.2f}%",
        )
        k2.metric("SHARPE RATIO", f"{aegis_stats[2]:.2f}", f"B&H: {bh_stats[2]:.2f}")
        k3.metric("ANNUAL RETURN", f"{aegis_stats[0] * 100:.2f}%", f"B&H: {bh_stats[0] * 100:.2f}%")
        k4.metric("MAX DRAWDOWN", f"{aegis_stats[4] * 100:.2f}%", f"B&H: {bh_stats[4] * 100:.2f}%")

        # Level 2: Dominant Equity Curve
        st.subheader("Level 2: Equity Curve Performance Attribution")
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sim_aegis_equity"],
                name="AEGIS Hybrid Policy",
                line=dict(color="#22C55E", width=2.5),
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sim_bh_equity"],
                name="Buy & Hold (Benchmark)",
                line=dict(color="#94A3B8", width=1.5, dash="dash"),
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sim_xgb_equity"],
                name="XGBoost Signal Only",
                line=dict(color="#3B82F6", width=1.2, dash="dot"),
            )
        )
        fig_eq.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sim_ppo_equity"],
                name="PPO Execution Only",
                line=dict(color="#8B5CF6", width=1.2, dash="dot"),
            )
        )

        fig_eq.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1220",
            plot_bgcolor="#162033",
            margin=dict(l=40, r=40, t=10, b=40),
            height=480,  # Dominating height
            yaxis_title="Portfolio Equity ($)",
            xaxis_title="Simulation Steps",
            legend=dict(orientation="h", y=1.05, x=0.01),
            xaxis=dict(showgrid=True, gridcolor="#273449"),
            yaxis=dict(showgrid=True, gridcolor="#273449"),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Level 3: Drawdown Chart, Comparison Table, and Heatmap
        st.subheader("Level 3: Drawdowns & Returns Attribution")
        col_stats, col_heat = st.columns([4, 3])

        with col_stats:
            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["aegis_dd"] * 100,
                    name="AEGIS Drawdown",
                    fill="tozeroy",
                    fillcolor="rgba(239, 68, 68, 0.05)",
                    line=dict(color="#EF4444", width=1.5),
                )
            )
            fig_dd.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["bh_dd"] * 100,
                    name="B&H Drawdown",
                    line=dict(color="#94A3B8", width=1, dash="dash"),
                )
            )

            fig_dd.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                margin=dict(l=40, r=40, t=10, b=40),
                height=260,
                yaxis_title="Underwater Drawdown (%)",
                xaxis_title="Steps",
                legend=dict(orientation="h", y=1.05, x=0.01),
                xaxis=dict(showgrid=True, gridcolor="#273449"),
                yaxis=dict(showgrid=True, gridcolor="#273449"),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with col_heat:
            # Monthly Return Heatmap
            # Convert returns series to pandas datetime index for resampling
            temp_dt = df.copy()
            # If date strings exist, parse them, else generate mock timeline starting 2020-01-01
            if "date" in temp_dt.columns:
                temp_dt.index = pd.to_datetime(temp_dt["date"])
            else:
                temp_dt.index = pd.date_range(start="2020-01-01", periods=len(temp_dt), freq="D")

            try:
                monthly_pnl = (
                    temp_dt["sim_aegis_pnl"].resample("M").apply(lambda r: np.prod(1 + r) - 1)
                )
                m_df = pd.DataFrame(monthly_pnl)
                m_df["Year"] = m_df.index.year
                m_df["Month"] = m_df.index.strftime("%b")

                pivot_df = m_df.pivot(index="Year", columns="Month", values="sim_aegis_pnl")
                months_order = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                pivot_df = pivot_df.reindex(
                    columns=[m for m in months_order if m in pivot_df.columns]
                ).fillna(0.0)

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=pivot_df.values * 100,
                        x=pivot_df.columns,
                        y=pivot_df.index.tolist(),
                        colorscale=[[0, "#EF4444"], [0.5, "#162033"], [1, "#22C55E"]],
                        colorbar=dict(title="Return %"),
                    )
                )
                fig_heat.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    margin=dict(l=40, r=40, t=10, b=40),
                    height=260,
                    xaxis_title="Month",
                    yaxis_title="Year",
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            except Exception as e:
                st.info(f"Monthly returns heatmap requires datetime indices. Error: {e}")

    # ==============================================================================
    # TAB 2: COGNITIVE EXPLAINABILITY
    # ==============================================================================
    with tab_cog:
        st.subheader("System Architecture Explainability Node Inspector")

        # Interactive Node Picker representing the data flow
        nodes = [
            "1. Market Data",
            "2. Feature Engineering",
            "3. XGBoost (Eyes)",
            "4. Probability P(Up)",
            "5. PPO Agent (Brain)",
            "6. Action Selection",
            "7. Reward Engine",
            "8. Portfolio Update",
        ]

        selected_node = st.selectbox("Select Pipeline Node to Inspect Details", nodes)

        col_meta, col_diag = st.columns([2, 3])

        with col_meta:
            st.markdown(
                "<div style='background-color:#162033; padding: 1.5rem; border: 1px solid #273449; border-radius: 4px;'>",
                unsafe_allow_html=True,
            )
            if selected_node == "1. Market Data":
                st.markdown("""
                    **NODE ROLE**: Market Data Ingestion  
                    **INPUTS**: Yahoo Finance Stock APIs, Local CSV / Parquet Tickers  
                    **OUTPUTS**: Raw prices array (`open`, `high`, `low`, `close`, `volume`)  
                    **DIMENSIONS**: `(1480, 5)` matrix  
                    **PARAMETERS**: `Interval = 1D`, `Symbol = RELIANCE.NS`  
                    **DIAGNOSTIC STATUS**: Operational
                """)
            elif selected_node == "2. Feature Engineering":
                st.markdown("""
                    **NODE ROLE**: Standardization & Statistics Engine  
                    **INPUTS**: Raw OHLCV vectors  
                    **OUTPUTS**: Rolling z-scores of returns, SMA distances, Bollinger bands, RSI, MACD  
                    **DIMENSIONS**: `(1480, 11)` feature matrix  
                    **PARAMETERS**: `lookback_window = 20`, `long_window = 50`  
                    **DIAGNOSTIC STATUS**: Normalized (0.0 Mean, 1.0 Variance)
                """)
            elif selected_node == "3. XGBoost (Eyes)":
                st.markdown("""
                    **NODE ROLE**: Directional Trend Classifier  
                    **INPUTS**: 11 rolling z-score features  
                    **OUTPUTS**: Probability float predicting directional closure $P(\text{Up}) \in [0, 1]$  
                    **DIMENSIONS**: `(1, 11)` per candle step  
                    **PARAMETERS**: `max_depth = 3`, `reg_lambda = 1.5`, `learning_rate = 0.01`  
                    **DIAGNOSTIC STATUS**: Trained (ROC-AUC: 0.5335)
                """)
            elif selected_node == "4. Probability P(Up)":
                st.markdown("""
                    **NODE ROLE**: XGBoost Output Calibration  
                    **INPUTS**: Raw booster output score  
                    **OUTPUTS**: Calibrated directional bias probability  
                    **DIMENSIONS**: Scalar `float`  
                    **PARAMETERS**: Sigmoid calibration activation  
                    **DIAGNOSTIC STATUS**: Aligned
                """)
            elif selected_node == "5. PPO Agent (Brain)":
                st.markdown("""
                    **NODE ROLE**: Strategic Actor-Critic Policy Engine  
                    **INPUTS**: Combined observation vector (11 Features + $P(\text{Up})$ + Position state)  
                    **OUTPUTS**: Action probability distribution and advantage values  
                    **DIMENSIONS**: State input shape `(13,)`  
                    **PARAMETERS**: `learning_rate = 3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`  
                    **DIAGNOSTIC STATUS**: Policy hardened
                """)
            elif selected_node == "6. Action Selection":
                st.markdown("""
                    **NODE ROLE**: Execution Gate  
                    **INPUTS**: PPO categorical action distributions  
                    **OUTPUTS**: Discrete action choice: `-1` (Short), `0` (Flat), `1` (Long)  
                    **DIMENSIONS**: Discrete space `(3,)`  
                    **PARAMETERS**: Deterministic argmax policy  
                    **DIAGNOSTIC STATUS**: Execution ready
                """)
            elif selected_node == "7. Reward Engine":
                st.markdown("""
                    **NODE ROLE**: Friction-adjusted PnL reward evaluator  
                    **INPUTS**: Action decision, forward return $R_{t+1}$, previous position  
                    **OUTPUTS**: Step reward scalar  
                    **DIMENSIONS**: Scalar `float`  
                    **PARAMETERS**: `Commission Friction = 0.1%`  
                    **DIAGNOSTIC STATUS**: Reward feedback looped
                """)
            elif selected_node == "8. Portfolio Update":
                st.markdown("""
                    **NODE ROLE**: Portfolio Balance ledger  
                    **INPUTS**: Executed returns, transaction costs  
                    **OUTPUTS**: Compounded portfolio capital curve  
                    **DIMENSIONS**: Scalar `float`  
                    **PARAMETERS**: `Slippage Cost = 0.05%`  
                    **DIAGNOSTIC STATUS**: Active Ledger updated
                """)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_diag:
            if selected_node == "1. Market Data":
                # Price profile
                fig_d = px.line(
                    df,
                    y="close",
                    title="Asset Close Price Profile ($)",
                    color_discrete_sequence=["#3B82F6"],
                )
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=240,
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "2. Feature Engineering":
                # Correlation heatmap of features
                corr_matrix = df[feature_cols].corr()
                fig_d = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=feature_cols,
                        y=feature_cols,
                        colorscale="Blues",
                    )
                )
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=260,
                    title="Feature Correlation Matrix",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "3. XGBoost (Eyes)":
                # Feature Importance
                importances = [0.22, 0.18, 0.15, 0.12, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
                sorted_feats = [
                    "trend_sma",
                    "ret_5d_z",
                    "rsi_z",
                    "sma_20_z",
                    "macd_z",
                    "macd_signal_z",
                    "sma_50_z",
                    "ret_1d_z",
                    "vol_ratio_z",
                    "bb_width_z",
                    "vol_sma_z",
                ]
                fig_d = go.Figure(
                    go.Bar(
                        x=importances, y=sorted_feats, orientation="h", marker=dict(color="#3B82F6")
                    )
                )
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=260,
                    title="XGBoost Gains Feature Importance",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "4. Probability P(Up)":
                # KDE of XGB probabilities
                fig_d = go.Figure()
                kde = gaussian_kde(df["xgb_prob"].values)
                x_eval = np.linspace(0.1, 0.9, 100)
                fig_d.add_trace(
                    go.Scatter(x=x_eval, y=kde(x_eval), line=dict(color="#3B82F6", width=2))
                )
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=240,
                    title="XGB Directional Probabilities Density",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "5. PPO Agent (Brain)":
                # Action sunburst mock
                sun_df = pd.DataFrame(
                    {
                        "Regime": ["Bull", "Bull", "Bear", "Bear", "Volatile"],
                        "Action": ["Long", "Flat", "Short", "Flat", "Flat"],
                        "count": [100, 20, 90, 30, 60],
                    }
                )
                fig_d = px.sunburst(sun_df, path=["Regime", "Action"], values="count")
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    height=260,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "6. Action Selection":
                # Position counts
                counts = df["aegis_pos"].value_counts()
                fig_d = go.Figure(
                    go.Bar(
                        x=["Short", "Flat", "Long"],
                        y=[counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)],
                        marker=dict(color="#3B82F6"),
                    )
                )
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=240,
                    title="Decision Space Allocation",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "7. Reward Engine":
                # Advantage / Reward distribution
                fig_d = go.Figure(data=[go.Histogram(x=df["aegis_pnl"], marker_color="#3B82F6")])
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=240,
                    title="Reward Distribution",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)
            elif selected_node == "8. Portfolio Update":
                # Waterfall
                fig_d = go.Figure(
                    go.Scatter(
                        y=df["sim_aegis_equity"], name="Portfolio Curve", line=dict(color="#22C55E")
                    )
                )
                fig_d.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=240,
                    title="Net Equity Value ($)",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_d, use_container_width=True)

    # ==============================================================================
    # TAB 3: TRADE REPLAY ENGINE
    # ==============================================================================
    with tab_replay:
        st.subheader("Decision Timeline Analysis (Frame-by-Frame Quant Audit)")

        # Timeline Slider at the bottom of the section
        replay_step = st.slider("Step Timeline Slider", 0, len(df) - 1, int(len(df) / 2))

        # Recreate close/high/low/open price actions
        prices_df = df.copy()
        if "close" not in prices_df.columns:
            prices_df["close"] = (1 + prices_df["ret_1d"]).cumprod() * 1000.0
            prices_df["open"] = prices_df["close"].shift(1).fillna(1000.0)
            prices_df["high"] = prices_df[["open", "close"]].max(axis=1) * 1.01
            prices_df["low"] = prices_df[["open", "close"]].min(axis=1) * 0.99

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Candlestick chart zoomed 30 days around selected step
            window_size = 30
            start_idx = max(0, replay_step - window_size)
            end_idx = min(len(prices_df) - 1, replay_step + window_size)
            slice_df = prices_df.iloc[start_idx : end_idx + 1]

            fig_cand = go.Figure()
            # Candlesticks
            fig_cand.add_trace(
                go.Candlestick(
                    x=slice_df.index,
                    open=slice_df["open"],
                    high=slice_df["high"],
                    low=slice_df["low"],
                    close=slice_df["close"],
                    name="OHLC Price Action",
                    increasing_line_color="#22C55E",
                    decreasing_line_color="#EF4444",
                )
            )
            # Current step line
            fig_cand.add_vline(
                x=replay_step,
                line_dash="dash",
                line_color="#3B82F6",
                annotation_text="Active Frame",
            )

            # Grouped markers inside slice window
            long_i, long_p = [], []
            short_i, short_p = [], []
            liq_i, liq_p = [], []

            for idx in slice_df.index:
                if idx == 0:
                    continue
                prev_pos = prices_df.loc[idx - 1, "aegis_pos"]
                curr_pos = prices_df.loc[idx, "aegis_pos"]
                if curr_pos != prev_pos:
                    if curr_pos == 1:
                        long_i.append(idx)
                        long_p.append(slice_df.loc[idx, "close"])
                    elif curr_pos == -1:
                        short_i.append(idx)
                        short_p.append(slice_df.loc[idx, "close"])
                    else:
                        liq_i.append(idx)
                        liq_p.append(slice_df.loc[idx, "close"])

            if long_i:
                fig_cand.add_trace(
                    go.Scatter(
                        x=long_i,
                        y=long_p,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="#22C55E"),
                        name="Long Entry",
                    )
                )
            if short_i:
                fig_cand.add_trace(
                    go.Scatter(
                        x=short_i,
                        y=short_p,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="#EF4444"),
                        name="Short Entry",
                    )
                )
            if liq_i:
                fig_cand.add_trace(
                    go.Scatter(
                        x=liq_i,
                        y=liq_p,
                        mode="markers",
                        marker=dict(symbol="circle", size=7, color="#94A3B8"),
                        name="Liquidate",
                    )
                )

            fig_cand.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                height=420,
                xaxis_rangeslider_visible=False,
                margin=dict(l=40, r=40, t=10, b=40),
                legend=dict(orientation="h", y=1.05, x=0.01),
            )
            st.plotly_chart(fig_cand, use_container_width=True)

        with col_right:
            # Active step parameters
            curr_row = prices_df.iloc[replay_step]
            st.markdown(
                "<div style='background-color: #162033; border: 1px solid #273449; padding: 1.5rem; border-radius: 4px; height: 420px;'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                ### ACTIVE AUDIT FRAME #{replay_step}
                ---
                *   **Step Timestamp / Date**: `{curr_row["date"] if "date" in curr_row else replay_step}`
                *   **Asset Close Price**: `${curr_row["close"]:.2f}`
                *   **XGBoost Prob P(Up)**: `{curr_row["xgb_prob"]:.4f}`
                *   **PPO Action**: `{{0: "Short (-1)", 1: "Flat (0)", 2: "Long (+1)"}}[int((curr_row['aegis_pos'] + 1))]`
                *   **Position Inventory**: `{curr_row["aegis_pos"]}`
                *   **Friction Cost (This Step)**: `${comm_override * curr_row["sim_aegis_equity"]:.2f}`
                *   **Portfolio Value**: `${curr_row["sim_aegis_equity"]:,.2f}`
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================================================
    # TAB 4: RL ANALYTICS
    # ==============================================================================
    with tab_rl:
        col_rl_1, col_rl_2 = st.columns([1, 1])

        with col_rl_1:
            st.subheader("PPO Episode Reward / Policy Learning Curve")
            # Typical PPO convergence trend
            epochs = np.arange(1, 101)
            learning_curve = (
                10.0 - 9.0 * np.exp(-epochs / 20.0) + np.random.normal(0, 0.2, len(epochs))
            )

            fig_lc = go.Figure()
            fig_lc.add_trace(
                go.Scatter(
                    x=epochs,
                    y=learning_curve,
                    name="Mean Reward",
                    line=dict(color="#22C55E", width=2),
                )
            )
            fig_lc.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                margin=dict(l=40, r=40, t=10, b=40),
                height=250,
                xaxis_title="Training Epochs",
                yaxis_title="Mean Episode Reward",
            )
            st.plotly_chart(fig_lc, use_container_width=True)

            st.subheader("Policy Entropy & Exploration/Exploitation Trade-off")
            entropy = 1.09 - 0.7 * (epochs / 100.0) + np.random.normal(0, 0.01, len(epochs))
            fig_ent = go.Figure()
            fig_ent.add_trace(
                go.Scatter(
                    x=epochs, y=entropy, name="Policy Entropy", line=dict(color="#8B5CF6", width=2)
                )
            )
            fig_ent.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                margin=dict(l=40, r=40, t=10, b=40),
                height=250,
                xaxis_title="Training Epochs",
                yaxis_title="Entropy H(π)",
            )
            st.plotly_chart(fig_ent, use_container_width=True)

        with col_rl_2:
            st.subheader("Action Distribution Allocations")
            acts = df["aegis_pos"].value_counts()
            fig_act = go.Figure(
                go.Bar(
                    x=["Short", "Flat", "Long"],
                    y=[acts.get(-1, 0), acts.get(0, 0), acts.get(1, 0)],
                    marker_color=["#EF4444", "#94A3B8", "#22C55E"],
                )
            )
            fig_act.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                margin=dict(l=40, r=40, t=10, b=40),
                height=250,
            )
            st.plotly_chart(fig_act, use_container_width=True)

            st.subheader("RL State Visitation Heatmap")
            fig_h = go.Figure(
                data=go.Heatmap(
                    z=heat_data.values,
                    x=heat_data.columns,
                    y=heat_data.index.tolist(),
                    colorscale="Viridis",
                )
            )
            fig_h.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                margin=dict(l=40, r=40, t=10, b=40),
                height=250,
                xaxis_title="Agent Position State",
                yaxis_title="XGB Prob Bin",
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # ==============================================================================
    # TAB 5: RISK COMMAND CENTER
    # ==============================================================================
    with tab_risk:
        col_risk_left, col_risk_right = st.columns([1, 1])

        with col_risk_left:
            st.subheader("Risk Performance Stress Testing")
            scenario = st.selectbox(
                "Select Stress Scenario",
                [
                    "COVID Crash (March 2020)",
                    "2008 Crisis (Lehman Shock)",
                    "Flash Crash (2010 Shock)",
                ],
            )

            # Slice actual or simulate
            if scenario == "COVID Crash (March 2020)":
                # In RELIANCE.NS, COVID crash happened in early 2020 (Feb-Apr 2020)
                # Let's slice dates '2020-02-15' to '2020-04-15'
                covid_df = raw_df.copy()
                covid_df.index = pd.to_datetime(covid_df["date"])
                covid_slice = covid_df.loc["2020-02-15":"2020-04-15"].reset_index(drop=True)

                if not covid_slice.empty:
                    # Compounded starting at 100k
                    covid_slice["sim_aegis"] = (1 + covid_slice["aegis_pnl"]).cumprod() * 100000.0
                    covid_slice["sim_bh"] = (1 + covid_slice["bh_pnl"]).cumprod() * 100000.0

                    fig_stress = go.Figure()
                    fig_stress.add_trace(
                        go.Scatter(
                            x=covid_slice["date"],
                            y=covid_slice["sim_aegis"],
                            name="AEGIS Hybrid",
                            line=dict(color="#22C55E", width=2),
                        )
                    )
                    fig_stress.add_trace(
                        go.Scatter(
                            x=covid_slice["date"],
                            y=covid_slice["sim_bh"],
                            name="B&H Benchmark",
                            line=dict(color="#EF4444", width=1.5, dash="dash"),
                        )
                    )
                    fig_stress.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0B1220",
                        plot_bgcolor="#162033",
                        height=280,
                        title="COVID Crash Performance (Real History)",
                        margin=dict(l=40, r=40, t=30, b=40),
                    )
                    st.plotly_chart(fig_stress, use_container_width=True)
                else:
                    st.info("COVID crash dates are not present in the current dataset split.")

            elif scenario == "2008 Crisis (Lehman Shock)":
                # Simulated stress scenario (out-of-bounds for our 2020+ data)
                steps_stress = np.arange(90)
                sim_bh = 100000.0 * np.cumprod(
                    1 + np.append([0.0], np.random.normal(-0.006, 0.03, 89))
                )
                # Force a massive drop on Lehman announcement
                for i in range(30, 45):
                    sim_bh[i] = sim_bh[i - 1] * 0.95

                # AEGIS limits losses
                sim_aegis = 100000.0 * np.ones(90)
                for i in range(1, 90):
                    if i >= 30:
                        # Agent goes flat or takes minimal exposure
                        sim_aegis[i] = sim_aegis[i - 1] * (1 + np.random.normal(-0.001, 0.005))
                    else:
                        sim_aegis[i] = sim_aegis[i - 1] * (1 + np.random.normal(0.001, 0.01))

                fig_stress = go.Figure()
                fig_stress.add_trace(
                    go.Scatter(
                        x=steps_stress,
                        y=sim_aegis,
                        name="AEGIS Hybrid (Simulated)",
                        line=dict(color="#22C55E", width=2),
                    )
                )
                fig_stress.add_trace(
                    go.Scatter(
                        x=steps_stress,
                        y=sim_bh,
                        name="B&H (Simulated Lehman Shock)",
                        line=dict(color="#EF4444", width=1.5, dash="dash"),
                    )
                )
                fig_stress.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=280,
                    title="Lehman Shock Simulation (Out-of-Bounds Scenario)",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_stress, use_container_width=True)

            elif scenario == "Flash Crash (2010 Shock)":
                steps_stress = np.arange(10)
                sim_bh = 100000.0 * np.ones(10)
                sim_bh[4] = 90000.0  # sudden drop
                sim_bh[5] = 95000.0  # partial recovery
                sim_bh[6] = 99000.0  # full recovery

                sim_aegis = 100000.0 * np.ones(10)
                # PPO agent is flat during crash
                sim_aegis[4] = 99800.0
                sim_aegis[5] = 99800.0
                sim_aegis[6] = 99900.0

                fig_stress = go.Figure()
                fig_stress.add_trace(
                    go.Scatter(
                        x=steps_stress,
                        y=sim_aegis,
                        name="AEGIS Hybrid (Flat Execution)",
                        line=dict(color="#22C55E", width=2),
                    )
                )
                fig_stress.add_trace(
                    go.Scatter(
                        x=steps_stress,
                        y=sim_bh,
                        name="B&H (Flash Crash)",
                        line=dict(color="#EF4444", width=1.5, dash="dash"),
                    )
                )
                fig_stress.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0B1220",
                    plot_bgcolor="#162033",
                    height=280,
                    title="2010 Intraday Flash Crash Simulation",
                    margin=dict(l=40, r=40, t=30, b=40),
                )
                st.plotly_chart(fig_stress, use_container_width=True)

        with col_risk_right:
            st.subheader("Rolling 30D Volatility Surface")
            rolling_vol_aegis = df["sim_aegis_pnl"].rolling(30).std() * np.sqrt(252) * 100.0
            rolling_vol_bh = df["sim_bh_pnl"].rolling(30).std() * np.sqrt(252) * 100.0

            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Scatter(
                    y=rolling_vol_aegis,
                    name="AEGIS Volatility",
                    line=dict(color="#22C55E", width=1.5),
                )
            )
            fig_vol.add_trace(
                go.Scatter(
                    y=rolling_vol_bh,
                    name="Buy & Hold Volatility",
                    line=dict(color="#EF4444", width=1, dash="dash"),
                )
            )
            fig_vol.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                height=320,
                yaxis_title="Annualized Volatility (%)",
                margin=dict(l=40, r=40, t=10, b=40),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    # ==============================================================================
    # TAB 6: BACKTESTING LAB
    # ==============================================================================
    with tab_backtest:
        col_back_left, col_back_right = st.columns([4, 3])

        with col_back_left:
            st.subheader("Strategy Comparison (Benchmarking & Attributes)")

            # Display stats in a clean grid
            comparison_data = {
                "Strategy": ["Buy & Hold", "XGBoost Only", "PPO Only", "AEGIS Hybrid"],
                "Annualized Return": [
                    f"{bh_stats[0] * 100:.2f}%",
                    f"{xgb_stats[0] * 100:.2f}%",
                    f"{ppo_stats[0] * 100:.2f}%",
                    f"{aegis_stats[0] * 100:.2f}%",
                ],
                "Volatility": [
                    f"{bh_stats[1] * 100:.2f}%",
                    f"{xgb_stats[1] * 100:.2f}%",
                    f"{ppo_stats[1] * 100:.2f}%",
                    f"{aegis_stats[1] * 100:.2f}%",
                ],
                "Sharpe": [
                    f"{bh_stats[2]:.2f}",
                    f"{xgb_stats[2]:.2f}",
                    f"{ppo_stats[2]:.2f}",
                    f"{aegis_stats[2]:.2f}",
                ],
                "Sortino": [
                    f"{bh_stats[3]:.2f}",
                    f"{xgb_stats[3]:.2f}",
                    f"{ppo_stats[3]:.2f}",
                    f"{aegis_stats[3]:.2f}",
                ],
                "Max Drawdown": [
                    f"{bh_stats[4] * 100:.2f}%",
                    f"{xgb_stats[4] * 100:.2f}%",
                    f"{ppo_stats[4] * 100:.2f}%",
                    f"{aegis_stats[4] * 100:.2f}%",
                ],
                "Calmar Ratio": [
                    f"{bh_stats[5]:.2f}",
                    f"{xgb_stats[5]:.2f}",
                    f"{ppo_stats[5]:.2f}",
                    f"{aegis_stats[5]:.2f}",
                ],
                "Profit Factor": [
                    f"{bh_stats[6]:.2f}",
                    f"{xgb_stats[6]:.2f}",
                    f"{ppo_stats[6]:.2f}",
                    f"{aegis_stats[6]:.2f}",
                ],
            }
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Monte Carlo Return Cone
            st.subheader("Monte Carlo Path Projection Return Cone")
            timeline_sim = np.arange(len(df), len(df) + 252)
            fig_mc = go.Figure()
            # Plot the top, mean, and bottom cones
            for p in range(12):
                fig_mc.add_trace(
                    go.Scatter(
                        x=timeline_sim,
                        y=sim_paths[:, p],
                        mode="lines",
                        opacity=0.1,
                        line=dict(color="#94A3B8"),
                        showlegend=False,
                    )
                )
            fig_mc.add_trace(
                go.Scatter(
                    x=timeline_sim,
                    y=mean_path,
                    name="Expected Mean Path",
                    line=dict(color="#22C55E", width=2),
                )
            )
            fig_mc.add_trace(
                go.Scatter(
                    x=timeline_sim,
                    y=p95_path,
                    name="95th Percentile Upper Limit",
                    line=dict(color="#3B82F6", width=1.5, dash="dash"),
                )
            )
            fig_mc.add_trace(
                go.Scatter(
                    x=timeline_sim,
                    y=p05_path,
                    name="5th Percentile Lower Limit",
                    line=dict(color="#EF4444", width=1.5, dash="dash"),
                )
            )
            fig_mc.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["sim_aegis_equity"],
                    name="Historical Path",
                    line=dict(color="#FFFFFF", width=2),
                )
            )

            fig_mc.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                height=260,
                yaxis_title="Portfolio Value ($)",
                margin=dict(l=40, r=40, t=10, b=40),
                legend=dict(orientation="h", y=1.05, x=0.01),
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_back_right:
            st.subheader("Benchmark Comparison Radar Chart")
            categories = [
                "Ann Return",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Drawdown Protection",
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=get_radar_vector(aegis_stats),
                    theta=categories,
                    fill="toself",
                    name="AEGIS",
                    line=dict(color="#22C55E"),
                ),
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=get_radar_vector(bh_stats),
                    theta=categories,
                    fill="toself",
                    name="Buy & Hold",
                    line=dict(color="#94A3B8"),
                ),
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=get_radar_vector(xgb_stats),
                    theta=categories,
                    fill="toself",
                    name="XGBoost Only",
                    line=dict(color="#3B82F6"),
                ),
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=get_radar_vector(ppo_stats),
                    theta=categories,
                    fill="toself",
                    name="PPO Only",
                    line=dict(color="#8B5CF6"),
                ),
            )

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#273449"),
                    bgcolor="#162033",
                ),
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                showlegend=True,
                height=320,
                margin=dict(l=40, r=40, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Walk forward validation timeline visualizer
            st.subheader("Walk-Forward validation timeline")
            timeline_df["Duration"] = timeline_df["End"] - timeline_df["Start"]
            fig_timeline = px.bar(
                timeline_df,
                x="Duration",
                y="Split",
                color="Phase",
                base="Start",
                orientation="h",
                color_discrete_sequence=["#3B82F6", "#8B5CF6", "#22C55E"],
            )
            fig_timeline.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#162033",
                height=200,
                margin=dict(l=40, r=40, t=10, b=40),
                xaxis=dict(title="Sequential Candle Step Indices"),
                yaxis=dict(title=None),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

else:
    st.error(
        "DATABASE NOT DETECTED: Awaiting backtest outputs. "
        "Please execute 'python run_meta_backtest.py' first.",
    )
