import numpy as np
import pandas as pd
import xgboost as xgb
from stable_baselines3 import PPO
import os

from src.envs.trading_env import TradingEnv
from src.consensus.hard_consensus import HardConsensus
from src.sentiment.sentiment_engine import SentimentEngine
from src.risk.risk_filters import RiskManager

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "data/RELIANCE.NS_processed.parquet"
XGB_PATH = "artifacts/xgb/xgb_directional.json"
PPO_PATH = "artifacts/ppo/ppo_directional"
SENTIMENT_PATH = "data/sentiment_daily.csv"
OUTPUT_PATH = "artifacts/backtests/hard_consensus_results.csv"

COMMISSION = 0.001  # 0.1% per trade

print("=== RUNNING CORRECTED HARD CONSENSUS BACKTEST ===")

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_parquet(DATA_PATH).reset_index()

feature_cols = [
    "ret_1d_z", "ret_5d_z", "sma_20_z", "sma_50_z",
    "rsi_z", "macd_z", "macd_signal_z",
    "bb_width_z", "vol_sma_z", "vol_ratio_z",
    "trend_sma"
]

# Clean data
df = df[np.isfinite(df[feature_cols]).all(axis=1)].reset_index(drop=True)

# ==================================================
# LOAD ENGINES
# ==================================================

# 1. XGBoost
booster = xgb.Booster()
booster.load_model(XGB_PATH)

# 2. PPO
env = TradingEnv(df)
model = PPO.load(PPO_PATH, env=env)

# 3. Sentiment
try:
    sent_df = pd.read_csv(SENTIMENT_PATH)
    if sent_df.empty:
        # Create dummy sentiment if file is empty
        sent_df = pd.DataFrame(columns=["date", "symbol", "sentiment_score"])
    sentiment_engine = SentimentEngine(sent_df)
except Exception as e:
    print(f"⚠️ Sentiment error: {e}. Using neutral sentiment.")
    sent_df = pd.DataFrame(columns=["date", "symbol", "sentiment_score"])
    sentiment_engine = SentimentEngine(sent_df)

# 4. Risk
risk_mgr = RiskManager(max_volatility=0.03, max_drawdown=0.10)

# ==================================================
# MAIN BACKTEST LOOP
# ==================================================
results = []
current_equity = 1.0
prev_position = 0

# Split point (MUST match training split)
split_idx = int(len(df) * 0.7)

for t in range(len(df) - 1):
    # --- Observation at t ---
    obs = df.loc[t, feature_cols].values.astype(np.float32)
    obs_reshape = obs.reshape(1, -1)
    
    # 1. XGB Signal
    dmat = xgb.DMatrix(obs_reshape, feature_names=feature_cols)
    xgb_prob = booster.predict(dmat)[0]
    xgb_sig = 0
    if xgb_prob > 0.60: xgb_sig = 1
    elif xgb_prob < 0.40: xgb_sig = -1
    
    # 2. PPO Signal
    ppo_act, _ = model.predict(obs, deterministic=True)
    ppo_sig = {0: -1, 1: 0, 2: 1}[int(ppo_act)]
    
    # 3. Sentiment Signal
    sent_sig = sentiment_engine.get_signal(pd.to_datetime(df.loc[t, "date"]), "RELIANCE.NS")
    
    # --- Consensus Decision ---
    final_pos = 0
    if xgb_sig == ppo_sig == sent_sig and xgb_sig != 0:
        final_pos = xgb_sig
    
    # --- Risk Filters ---
    # Apply volatility gate
    recent_rets = [r["pnl"] for r in results[-20:]] if len(results) >= 20 else []
    if not risk_mgr.volatility_gate(np.array(recent_rets)):
        final_pos = 0
        
    # --- Execution & Costs ---
    cost = COMMISSION if final_pos != prev_position else 0
    
    # --- Earn return of t+1 ---
    next_ret = df.loc[t + 1, "ret_1d"]
    step_pnl = (final_pos * next_ret) - cost
    
    # Update Equity (Compounded)
    current_equity *= (1 + step_pnl)
    
    # Track results
    period = "IS" if t < split_idx else "OOS"
    
    results.append({
        "equity": current_equity,
        "position": final_pos,
        "xgb_signal": xgb_sig,
        "ppo_signal": ppo_sig,
        "sentiment_signal": sent_sig,
        "pnl": step_pnl,
        "period": period
    })
    
    prev_position = final_pos

# ==================================================
# FINALIZE
# ==================================================
res_df = pd.DataFrame(results)

# Calculate Drawdown
res_df["peak"] = res_df["equity"].cummax()
res_df["drawdown"] = (res_df["equity"] / res_df["peak"]) - 1

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
res_df.to_csv(OUTPUT_PATH, index=False)

print(f"DONE: Corrected Backtest Complete. Results saved to {OUTPUT_PATH}")
print(f"Final Equity: {current_equity:.4f}")
print(f"Max Drawdown: {res_df['drawdown'].min():.4f}")
