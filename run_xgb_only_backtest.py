import numpy as np
import pandas as pd
import xgboost as xgb

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "data/RELIANCE.NS_processed.parquet"
XGB_MODEL_PATH = "artifacts/xgb/xgb_directional.json"

print("=== XGBOOST ONLY BACKTEST ===")

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
booster = xgb.Booster()
booster.load_model(XGB_MODEL_PATH)

# ==================================================
# BACKTEST
# ==================================================
returns = []
actions = 0

for t in range(len(df) - 1):
    X = df.loc[[t], feature_cols]
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    prob = booster.predict(dmat)[0]

    # Direction only
    if prob > 0.55:
        action = 1
    elif prob < 0.45:
        action = -1
    else:
        action = 0

    ret = df.loc[t + 1, "ret_1d"]
    pnl = action * ret

    if action != 0:
        actions += 1

    returns.append(pnl)

# ==================================================
# METRICS
# ==================================================
returns = np.array(returns)
equity = np.cumsum(returns)
max_dd = np.min(equity - np.maximum.accumulate(equity))

print(f"Cumulative Return : {equity[-1]:.4f}")
print(f"Max Drawdown      : {max_dd:.4f}")
print(f"Trades Taken      : {actions}")
