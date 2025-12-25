import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from src.envs.trading_env import TradingEnv

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "data/RELIANCE.NS_processed.parquet"
PPO_MODEL_PATH = "artifacts/ppo/ppo_directional"

print("=== PPO ONLY BACKTEST ===")

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_parquet(DATA_PATH).reset_index(drop=True)

feature_cols = [
    "ret_1d_z", "ret_5d_z",
    "sma_20_z", "sma_50_z",
    "rsi_z",
    "macd_z", "macd_signal_z",
    "bb_width_z",
    "vol_sma_z", "vol_ratio_z",
    "trend_sma"
]

df = df[np.isfinite(df[feature_cols]).all(axis=1)].reset_index(drop=True)

# ==================================================
# LOAD MODEL
# ==================================================
env = TradingEnv(df)
model = PPO.load(PPO_MODEL_PATH, env=env)

# ==================================================
# BACKTEST
# ==================================================
obs, _ = env.reset()
returns = []
actions = 0

for t in range(len(df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    mapped_action = {0: -1, 1: 0, 2: 1}[action]

    ret = df.loc[t + 1, "ret_1d"]
    pnl = mapped_action * ret

    if mapped_action != 0:
        actions += 1

    returns.append(pnl)

    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

# ==================================================
# METRICS
# ==================================================
returns = np.array(returns)
equity = np.cumsum(returns)
max_dd = np.min(equity - np.maximum.accumulate(equity))

print(f"Cumulative Return : {equity[-1]:.4f}")
print(f"Max Drawdown      : {max_dd:.4f}")
print(f"Trades Taken      : {actions}")
