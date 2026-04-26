import numpy as np
import pandas as pd
import xgboost as xgb
from stable_baselines3 import PPO
import os

from src.envs.trading_env import TradingEnv

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "data/RELIANCE.NS_processed.parquet"
XGB_PATH = "artifacts/xgb/xgb_directional.json"
PPO_PATH = "artifacts/ppo/ppo_meta_policy"
OUTPUT_PATH = "artifacts/backtests/meta_policy_results.csv"

COMMISSION = 0.001  # 0.1% per trade

print("=== RUNNING META-POLICY BACKTEST ===")

# ==================================================
# LOAD DATA & PRECOMPUTE XGB
# ==================================================
df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

feature_cols = [
    "ret_1d_z", "ret_5d_z", "sma_20_z", "sma_50_z",
    "rsi_z", "macd_z", "macd_signal_z",
    "bb_width_z", "vol_sma_z", "vol_ratio_z",
    "trend_sma"
]

# Clean data
df = df[np.isfinite(df[feature_cols]).all(axis=1)].reset_index(drop=True)

# LOAD XGB
booster = xgb.Booster()
booster.load_model(XGB_PATH)

# PRECOMPUTE PROBABILITIES
dmat = xgb.DMatrix(df[feature_cols])
df["xgb_prob"] = booster.predict(dmat)

# ==================================================
# LOAD PPO META-POLICY
# ==================================================
env = TradingEnv(df)
model = PPO.load(PPO_PATH, env=env)

# ==================================================
# MAIN BACKTEST LOOP
# ==================================================
results = []
current_equity = 1.0
prev_position = 0

# Split points
split_1 = int(len(df) * 0.6)
split_2 = int(len(df) * 0.8)

for t in range(len(df) - 1):
    # --- Observation at t ---
    obs = env.get_observation(t, prev_position, df.loc[t, "xgb_prob"])
    
    # --- PPO Decision ---
    ppo_act, _ = model.predict(obs, deterministic=True)
    final_pos = {0: -1, 1: 0, 2: 1}[int(ppo_act)]
        
    # --- Execution & Costs ---
    cost = COMMISSION if final_pos != prev_position else 0
    
    # --- Earn return of t+1 ---
    next_ret = df.loc[t + 1, "ret_1d"]
    step_pnl = (final_pos * next_ret) - cost
    
    # Update Equity (Compounded)
    current_equity *= (1 + step_pnl)
    
    # Track results
    if t < split_1:
        period = "IS_XGB_TRAIN"
    elif t < split_2:
        period = "IS_PPO_TRAIN"
    else:
        period = "OOS_TEST"
    
    results.append({
        "date": df.loc[t, "date"] if "date" in df.columns else t,
        "equity": current_equity,
        "position": final_pos,
        "xgb_prob": df.loc[t, "xgb_prob"],
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

print(f"DONE: Meta-Policy Backtest Complete. Results saved to {OUTPUT_PATH}")

# Print metrics by period
for period in ["IS_XGB_TRAIN", "IS_PPO_TRAIN", "OOS_TEST"]:
    subset = res_df[res_df["period"] == period]
    if not subset.empty:
        cum_ret = (subset["equity"].iloc[-1] / subset["equity"].iloc[0]) - 1
        sharpe = np.sqrt(252) * subset["pnl"].mean() / subset["pnl"].std() if subset["pnl"].std() != 0 else 0
        mdd = subset["drawdown"].min()
        exposure = (subset["position"] != 0).mean()
        print(f"[{period}] Cum Ret: {cum_ret:.2%}, Sharpe: {sharpe:.2f}, MaxDD: {mdd:.2%}, Exposure: {exposure:.2%}")

print(f"Final Equity (Overall): {current_equity:.4f}")
