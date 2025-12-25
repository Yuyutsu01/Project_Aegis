import pandas as pd
import os
from src.models.xgb_model import XGBDirectionalModel
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "data/RELIANCE.NS_processed.parquet"
ARTIFACT_DIR = "artifacts/xgb"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=== XGBOOST TRAINING (SAFE MODE) ===")

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
# TIME-BASED SPLIT
# --------------------------------------------------
split_1 = int(len(df) * 0.7)
split_2 = int(len(df) * 0.85)

X_train, y_train = X.iloc[:split_1], y.iloc[:split_1]
X_val, y_val     = X.iloc[split_1:split_2], y.iloc[split_1:split_2]
X_test, y_test   = X.iloc[split_2:], y.iloc[split_2:]

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
model = XGBDirectionalModel()
model.train(X_train, y_train, X_val, y_val)

# --------------------------------------------------
# EVALUATE
# --------------------------------------------------
proba = model.predict_proba(X_test)
pred = (proba > 0.5).astype(int)

acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test ROC-AUC : {auc:.4f}")

# --------------------------------------------------
# BASELINE CHECK
# --------------------------------------------------
if auc < 0.52:
    raise RuntimeError("Model does not beat random baseline. STOP.")

print("✅ XGBoost PASSED baseline sanity check")

# -----------------------------------
# SAVE MODEL (OPTIONAL BUT SAFE)
# -----------------------------------
MODEL_PATH = "artifacts/xgb/xgb_directional.json"
model.model.get_booster().save_model(MODEL_PATH)


print(f"✅ XGBoost model saved at: {MODEL_PATH}")
