import pandas as pd
import os
import numpy as np
from src.models.xgb_model import XGBDirectionalModel
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = "data/RELIANCE.NS_processed.parquet"
ARTIFACT_DIR = "artifacts/xgb"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=== XGBOOST TRAINING (META-POLICY MODE) ===")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

# --------------------------------------------------
# TARGET
# --------------------------------------------------
df["target"] = (df["ret_1d"].shift(-1) > 0).astype(int)
df = df.dropna()

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------
feature_cols = [c for c in df.columns if c.endswith("_z")] + ["trend_sma"]

X = df[feature_cols]
y = df["target"]

# --------------------------------------------------
# SPLIT FOR META-POLICY (60% XGB, 20% PPO, 20% TEST)
# --------------------------------------------------
split_1 = int(len(df) * 0.6)

X_train, y_train = X.iloc[:split_1], y.iloc[:split_1]

# --------------------------------------------------
# TIME-SERIES CROSS VALIDATION
# --------------------------------------------------
tscv = TimeSeriesSplit(n_splits=3)
aucs = []

for train_idx, val_idx in tscv.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_v, y_v = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    cv_model = XGBDirectionalModel()
    cv_model.train(X_tr, y_tr, X_v, y_v)
    
    proba = cv_model.predict_proba(X_v)
    aucs.append(roc_auc_score(y_v, proba))

print(f"CV ROC-AUC scores: {aucs}")
print(f"Mean CV ROC-AUC: {np.mean(aucs):.4f}")

# --------------------------------------------------
# FINAL MODEL TRAINING
# --------------------------------------------------
# Train on full 60% for the final model to be used by PPO
model = XGBDirectionalModel()
# using a small dummy validation set just for early stopping
split_val = int(len(X_train) * 0.9)
model.train(X_train.iloc[:split_val], y_train.iloc[:split_val], X_train.iloc[split_val:], y_train.iloc[split_val:])

# -----------------------------------
# SAVE MODEL
# -----------------------------------
MODEL_PATH = "artifacts/xgb/xgb_directional.json"
model.model.get_booster().save_model(MODEL_PATH)

print(f"DONE: XGBoost model saved at: {MODEL_PATH}")
