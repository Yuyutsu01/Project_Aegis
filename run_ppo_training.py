import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.trading_env import TradingEnv
from src.models.ppo_agent import build_ppo

DATA_PATH = "data/RELIANCE.NS_processed.parquet"

print("=== PPO TRAINING (DIRECTION ONLY, SAFE MODE) ===")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

# --------------------------------------------------
# HARD FILTER: REMOVE UNSTABLE ROWS (CRITICAL)
# --------------------------------------------------
feature_cols = [c for c in df.columns if c.endswith("_z")] + ["trend_sma"]

mask = np.isfinite(df[feature_cols]).all(axis=1)
df = df.loc[mask].reset_index(drop=True)

print(f"Filtered data rows: {len(df)}")

# --------------------------------------------------
# TIME-BASED SPLIT
# --------------------------------------------------
train_cut = int(len(df) * 0.7)
train_df = df.iloc[:train_cut].copy()
test_df = df.iloc[train_cut:].copy()

# --------------------------------------------------
# TRAINING ENV
# --------------------------------------------------
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
model = build_ppo(train_env)

# IMPORTANT: Fewer steps for stability
model.learn(total_timesteps=10_000)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
test_env = TradingEnv(test_df)
obs, _ = test_env.reset()

done = False
cum_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    cum_reward += reward
    done = terminated or truncated

print(f"PPO Test Cumulative Reward: {cum_reward:.6f}")

model.save("artifacts/ppo/ppo_directional")

