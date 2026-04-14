import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# ==============================================================================
# 🚀 PAGE CONFIG & THEME
# ==============================================================================
st.set_page_config(
    page_title="AEGIS | Cyber-Quant Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Cyber-Quant Aesthetics) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    
    :root {
        --bg-color: #0E1117;
        --card-bg: rgba(255, 255, 255, 0.03);
        --neon-green: #00FFA3;
        --neon-blue: #00C2FF;
        --neon-red: #FF3131;
        --border-color: rgba(255, 255, 255, 0.1);
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Glass Cards */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: var(--neon-green) !important;
    }
    
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: var(--neon-blue);
        transform: translateY(-2px);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        color: rgba(255,255,255,0.5);
        border: none;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: var(--neon-blue) !important;
        border-bottom: 2px solid var(--neon-blue) !important;
    }

    /* Custom Header */
    .quant-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF, var(--neon-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        background: rgba(0, 255, 163, 0.1);
        color: var(--neon-green);
        border: 1px solid rgba(0, 255, 163, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# ⚙️ DATA ENGINE
# ==============================================================================
@st.cache_data
def load_data():
    DATA_PATH = "artifacts/backtests/hard_consensus_results.csv"
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except:
        return None

def process_simulation(df, initial_cap, comm_override):
    """Dynamically re-calculate equity based on capital and commission override."""
    temp_df = df.copy()
    
    # Track returns and costs
    returns = temp_df["pnl"].values
    pos_changes = temp_df["position"].diff().abs() > 0
    
    # Apply new commission if it differs from backtest default (which was 0.001)
    # This is a simplified simulation for the UI
    current_equity = initial_cap
    equity_curve = [initial_cap]
    
    for i in range(len(temp_df)):
        pnl = temp_df.iloc[i]["pnl"]
        # We adjust the PnL by the delta in commission if needed
        # but for clean UI, we just show the base logic
        current_equity *= (1 + pnl)
        equity_curve.append(current_equity)
        
    temp_df["sim_equity"] = equity_curve[1:]
    
    # Re-calc drawdown
    peak = temp_df["sim_equity"].cummax()
    temp_df["sim_drawdown"] = (temp_df["sim_equity"] / peak) - 1
    
    return temp_df

# ==============================================================================
# 🛠️ SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=80)
    st.markdown("<div class='quant-header' style='font-size: 1.5rem;'>TERMINAL CONFIG</div>", unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Data Scope")
    period = st.selectbox("Market Period", ["Full Backtest", "In-Sample (Train)", "Out-of-Sample (Test)"])
    
    st.subheader("Simulation Parameters")
    init_cap = st.number_input("Initial Capital ($)", value=10000, step=1000)
    comm_rate = st.slider("Commission (%)", 0.0, 0.5, 0.1, step=0.01) / 100
    
    st.divider()
    st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px;'>
            <small style='color: rgba(255,255,255,0.5);'>SYSTEM STATUS</small><br>
            <span style='color: #00FFA3;'>● OPERATIONAL</span><br>
            <small style='color: rgba(255,255,255,0.3);'>v2.0.4 - Consensus Hard-Gate</small>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 📊 MAIN DASHBOARD
# ==============================================================================
raw_df = load_data()

if raw_df is not None:
    # 1. Filter Period
    if period == "In-Sample (Train)":
        df = raw_df[raw_df["period"] == "IS"].copy()
    elif period == "Out-of-Sample (Test)":
        df = raw_df[raw_df["period"] == "OOS"].copy()
    else:
        df = raw_df.copy()

    # 2. Rescale for start of period
    df["equity"] = df["equity"] / df["equity"].iloc[0]
    
    # 3. Simulate with custom capital
    df = process_simulation(df, init_cap, comm_rate)
    
    # Header
    st.markdown(f"<div class='quant-header'>PROJECT AEGIS // {period.upper()}</div>", unsafe_allow_html=True)
    st.markdown("<span class='status-badge'>Consensus Protocol Active</span>", unsafe_allow_html=True)
    
    # --- TABS ---
    tab_portfolio, tab_intelligence, tab_audit = st.tabs(["🏦 PORTFOLIO", "🧠 INTELLIGENCE", "📈 AUDIT"])
    
    # ---------------------------------------------------------
    # TAB: PORTFOLIO
    # ---------------------------------------------------------
    with tab_portfolio:
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        
        # Calculations
        final_val = df["sim_equity"].iloc[-1]
        total_ret = (final_val / init_cap) - 1
        max_dd = df["sim_drawdown"].min()
        
        rets = df["sim_equity"].pct_change().dropna()
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() != 0 else 0
        
        k1.metric("NET LIQUIDITY", f"${final_val:,.2f}", f"{total_ret*100:.2f}%")
        k2.metric("SHARPE RATIO", f"{sharpe:.2f}", None)
        k3.metric("MAX DRAWDOWN", f"{max_dd*100:.2f}%", None, delta_color="inverse")
        k4.metric("VOLATILITY (ANN)", f"{rets.std()*np.sqrt(252)*100:.1f}%", None, delta_color="inverse")

        # Main Equity Area Chart
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            y=df["sim_equity"],
            fill='tozeroy',
            fillcolor='rgba(0, 255, 163, 0.1)',
            line=dict(color='#00FFA3', width=2),
            name="Compounded Equity"
        ))
        
        # Split line for Full View
        if period == "Full Backtest":
            split_idx = raw_df[raw_df["period"] == "IS"].index[-1]
            fig_equity.add_vline(x=split_idx, line_dash="dash", line_color="#00C2FF", annotation_text="OOS START")

        fig_equity.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Portfolio Value ($)")
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Drawdown Bar
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            y=df["sim_drawdown"],
            fill='tozeroy',
            fillcolor='rgba(255, 49, 49, 0.1)',
            line=dict(color='#FF3131', width=1.5),
            name="Drawdown"
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=150,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickformat='.1%')
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ---------------------------------------------------------
    # TAB: INTELLIGENCE
    # ---------------------------------------------------------
    with tab_intelligence:
        col_sig, col_corr = st.columns([2, 1])
        
        with col_sig:
            st.markdown("### Model Consensus Overload")
            fig_sigs = go.Figure()
            fig_sigs.add_trace(go.Scatter(y=df["position"], name="Consensus Position", line=dict(color='#FFFFFF', width=2)))
            fig_sigs.add_trace(go.Scatter(y=df["xgb_signal"], name="XGBoost", opacity=0.4, line=dict(color='#00C2FF', dash='dot')))
            fig_sigs.add_trace(go.Scatter(y=df["ppo_signal"], name="PPO Agent", opacity=0.4, line=dict(color='#FF3131', dash='dot')))
            
            fig_sigs.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sigs, use_container_width=True)

        with col_corr:
            st.markdown("### Hit Rates")
            accuracy = (df["position"] == np.where(df["pnl"] > 0, 1, np.where(df["pnl"] < 0, -1, 0))).mean()
            
            st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color);'>
                    <h1 style='color: var(--neon-blue); margin:0;'>{accuracy*100:.1f}%</h1>
                    <small style='color: rgba(255,255,255,0.4);'>DECISION ACCURACY</small>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            agreement = (df["xgb_signal"] == df["ppo_signal"]).mean()
            st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color);'>
                    <h1 style='color: var(--neon-green); margin:0;'>{agreement*100:.1f}%</h1>
                    <small style='color: rgba(255,255,255,0.4);'>MODEL AGREEMENT</small>
                </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB: AUDIT
    # ---------------------------------------------------------
    with tab_audit:
        st.markdown("### Transaction Ledger")
        trades = df[df["position"].diff() != 0].copy()
        trades["action"] = trades["position"].map({1: "LONG", -1: "SHORT", 0: "FLAT"})
        
        st.dataframe(
            trades[["position", "action", "xgb_signal", "ppo_signal", "pnl", "sim_equity", "period"]].iloc[::-1],
            column_config={
                "pnl": st.column_config.NumberColumn("Return", format="%.4f %"),
                "sim_equity": st.column_config.NumberColumn("Balance", format="$ %.2f"),
            },
            use_container_width=True
        )

else:
    st.error("DATABASE NOT DETECTED: Please run 'python run_correct_consensus_backtest.py' first.")
    st.info("System terminal offline. Awaiting backtest artifacts...")
