import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.trading_env import TradingEnv
from src.models.ppo_agent import build_ppo
import xgboost as xgb

DATA_PATH = "data/RELIANCE.NS_processed.parquet"
XGB_PATH = "artifacts/xgb/xgb_directional.json"

print("=== PPO TRAINING (META-POLICY MODE) ===")

# --------------------------------------------------
# LOAD DATA & XGBOOST MODEL
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

feature_cols = [c for c in df.columns if c.endswith("_z")] + ["trend_sma"]
mask = np.isfinite(df[feature_cols]).all(axis=1)
df = df.loc[mask].reset_index(drop=True)

booster = xgb.Booster()
booster.load_model(XGB_PATH)

# PRECOMPUTE XGB PROBABILITIES FOR META-POLICY
dmat = xgb.DMatrix(df[feature_cols])
df["xgb_prob"] = booster.predict(dmat)

print(f"Filtered data rows: {len(df)}")

# --------------------------------------------------
# TIME-BASED SPLIT (60% XGB, 20% PPO Train, 20% PPO Test)
# --------------------------------------------------
split_1 = int(len(df) * 0.6)
split_2 = int(len(df) * 0.8)

# PPO trains on the validation set of XGBoost to learn how to handle unseen XGB signals
train_df = df.iloc[split_1:split_2].copy().reset_index(drop=True)
test_df = df.iloc[split_2:].copy().reset_index(drop=True)

# --------------------------------------------------
# TRAINING ENV
# --------------------------------------------------
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
model = build_ppo(train_env)

# Train PPO
model.learn(total_timesteps=100_000)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
test_env = TradingEnv(test_df)
obs, _ = test_env.reset()

done = False
cum_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    step_result = test_env.step(action)
    # Handle both new (5-tuple) and old (4-tuple) gym API just in case
    if len(step_result) == 5:
        obs, reward, terminated, truncated, _ = step_result
        done = terminated or truncated
    else:
        obs, reward, done, _ = step_result
    cum_reward += reward

print(f"PPO Test Cumulative Reward (OOS): {cum_reward:.6f}")

model.save("artifacts/ppo/ppo_meta_policy")

