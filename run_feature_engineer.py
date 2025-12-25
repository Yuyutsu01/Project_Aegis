import os
import pandas as pd
from src.feature_engineer import FeatureEngineer

SYMBOLS = ["RELIANCE.NS", "TCS.NS"]
DATA_DIR = "data"

engineer = FeatureEngineer()
print("=== RUNNING FEATURE ENGINEERING (FINAL, MULTIINDEX-SAFE) ===")

for symbol in SYMBOLS:
    hist_path = os.path.join(DATA_DIR, f"{symbol}_historical.parquet")
    out_path = os.path.join(DATA_DIR, f"{symbol}_processed.parquet")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Missing historical file: {hist_path}")

    print(f"\nProcessing {symbol}...")

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------
    df = pd.read_parquet(hist_path)

    # --------------------------------------------------
    # FLATTEN YFINANCE MULTIINDEX (CRITICAL FIX)
    # --------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # --------------------------------------------------
    # CANONICALIZE COLUMN NAMES
    # --------------------------------------------------
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # --------------------------------------------------
    # REMOVE DUPLICATE COLUMNS
    # --------------------------------------------------
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # --------------------------------------------------
    # ENFORCE REQUIRED SCHEMA
    # --------------------------------------------------
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns in {symbol}: {missing}")

    # --------------------------------------------------
    # INDEX SANITY
    # --------------------------------------------------
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # --------------------------------------------------
    # SYMBOL COLUMN
    # --------------------------------------------------
    df["symbol"] = symbol

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    df_feat = engineer.calculate_features(df)
    df_final = engineer.normalize_continuous(df_feat)

    # --------------------------------------------------
    # FINAL SANITY CHECK
    # --------------------------------------------------
    if df_final.tail(30).isnull().any().any():
        raise RuntimeError("NaNs detected in recent rows")

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------
    df_final.to_parquet(out_path)
    print(f"Saved {symbol}_processed.parquet | Shape: {df_final.shape}")

print("\n🎉 FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
