import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Symbol-safe, index-safe feature engineering.
    Uses transform() everywhere to avoid MultiIndex bugs.
    """

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ===============================
        # RETURNS
        # ===============================
        df["ret_1d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(1)
        )

        df["ret_5d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(5)
        )

        # ===============================
        # MOVING AVERAGES
        # ===============================
        df["sma_20"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        df["sma_50"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(50, min_periods=1).mean()
        )

        # ===============================
        # RSI (NUMERICALLY SAFE)
        # ===============================
        delta = df.groupby("symbol")["close"].diff()

        gain = delta.clip(lower=0).groupby(df["symbol"]).transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )

        loss = (-delta.clip(upper=0)).groupby(df["symbol"]).transform(
            lambda x: x.rolling(14, min_periods=1).mean()
        )

        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # ===============================
        # MACD (TRANSFORM — NO MULTIINDEX)
        # ===============================
        ema12 = df.groupby("symbol")["close"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )

        ema26 = df.groupby("symbol")["close"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )

        df["macd"] = ema12 - ema26

        df["macd_signal"] = df.groupby("symbol")["macd"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )

        # ===============================
        # BOLLINGER BAND WIDTH
        # ===============================
        mid = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        std = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )

        df["bb_width"] = (2 * std) / mid

        # ===============================
        # VOLUME FEATURES (CRITICAL FIX)
        # ===============================
        df["vol_sma"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        # volume is GUARANTEED Series because runner deduplicates columns
        df["vol_ratio"] = df["volume"] / df["vol_sma"]

        # ===============================
        # BINARY TREND SIGNAL
        # ===============================
        df["trend_sma"] = (df["sma_20"] > df["sma_50"]).astype(int)

        return df

    def normalize_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        base_cols = ["open", "high", "low", "close", "volume", "symbol"]
        binary_cols = ["trend_sma"]

        continuous_cols = [
            c for c in df.columns
            if c not in base_cols + binary_cols
        ]

        for col in continuous_cols:
            mean = df.groupby("symbol")[col].transform(
                lambda x: x.rolling(60, min_periods=1).mean()
            )

            std = df.groupby("symbol")[col].transform(
                lambda x: x.rolling(60, min_periods=1).std()
            ).replace(0, 1e-10)

            df[f"{col}_z"] = (df[col] - mean) / std

        return df
