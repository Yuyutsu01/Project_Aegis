import os
import pandas as pd
from src.data_ingestor import DataIngestor

# -----------------------------------
# CONFIG
# -----------------------------------
SYMBOLS = ["RELIANCE.NS", "TCS.NS"]
DATA_DIR = "data"
START_DATE = "2020-01-01"

os.makedirs(DATA_DIR, exist_ok=True)

print("=== RUNNING DATA INGESTION (FINAL) ===")

ingestor = DataIngestor(
    symbols=SYMBOLS,
    start_date=START_DATE
)

data = ingestor.fetch_historical()

for symbol, df in data.items():
    # -------------------------------
    # ENFORCE CANONICAL SCHEMA
    # -------------------------------
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Keep only required columns
    df = df[["open", "high", "low", "close", "volume"]]

    # Enforce datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Add symbol column
    df["symbol"] = symbol

    out_path = os.path.join(DATA_DIR, f"{symbol}_historical.parquet")
    df.to_parquet(out_path)

    print(f"Saved {symbol}_historical.parquet | Rows: {len(df)}")

print("\n✅ DATA INGESTION COMPLETED SUCCESSFULLY")
