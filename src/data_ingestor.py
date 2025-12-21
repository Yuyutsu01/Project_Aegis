import yfinance as yf
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

class DataIngestor:
    def __init__(self, symbols, start_date="2020-01-01"):
        self.symbols = symbols
        self.start_date = start_date

    def fetch_historical(self) -> Dict[str, pd.DataFrame]:
        data = {}

        for symbol in self.symbols:
            print(f"Fetching {symbol}...")
            try:
                df = yf.download(
                    symbol,
                    start=self.start_date,
                    progress=False,
                    auto_adjust=False
                )

                if df.empty:
                    print(f"⚠️ No data for {symbol}")
                    continue

                # SAFE column mapping
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })

                # HARD RULE: drop adjusted close to avoid corporate-action leakage
                df = df[["open", "high", "low", "close", "volume"]]

                # Enforce datetime index
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df.index.name = "date"

                df["symbol"] = symbol
                data[symbol] = df

            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")

        return data

    @staticmethod
    def basic_validation(df: pd.DataFrame):
        issues = []

        if not df.index.is_monotonic_increasing:
            issues.append("Index not monotonic increasing")

        if not df.index.is_unique:
            issues.append("Duplicate timestamps")

        if df.isnull().any().any():
            issues.append("NaN values present")

        if (df["volume"] <= 0).any():
            issues.append("Zero or negative volume")

        return issues if issues else ["OK"]
