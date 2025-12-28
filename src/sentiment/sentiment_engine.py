
import pandas as pd
from typing import Optional

class SentimentEngine:
    """
    Deterministic daily sentiment engine.
    Acts as a veto signal only (no standalone trading).
    """

    POS_THRESHOLD = 0.10
    NEG_THRESHOLD = -0.10

    def __init__(self, sentiment_df: pd.DataFrame):
        """
        sentiment_df must contain:
        ['date', 'symbol', 'sentiment_score']
        """
        required_cols = {"date", "symbol", "sentiment_score"}
        if not required_cols.issubset(sentiment_df.columns):
            raise ValueError(f"Sentiment data missing columns: {required_cols}")

        self.df = sentiment_df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.sort_values(["symbol", "date"], inplace=True)

    def get_signal(
        self,
        date: pd.Timestamp,
        symbol: str
    ) -> int:
        """
        Returns sentiment signal for a given date and symbol.
        Output ∈ {-1, 0, +1}
        """

        row = self.df[
            (self.df["symbol"] == symbol) &
            (self.df["date"] == date)
        ]

        if row.empty:
            return 1  # No sentiment → neutral veto

        score = float(row["sentiment_score"].iloc[0])

        if score > self.POS_THRESHOLD:
            return 1
        elif score < self.NEG_THRESHOLD:
            return -1
        else:
            return 0

    def batch_signals(self) -> pd.DataFrame:
        """
        Returns a dataframe with an added 'sentiment_signal' column.
        Useful for backtesting / diagnostics.
        """

        df = self.df.copy()

        def map_score(s):
            if s > self.POS_THRESHOLD:
                return 1
            elif s < self.NEG_THRESHOLD:
                return -1
            else:
                return 0

        df["sentiment_signal"] = df["sentiment_score"].apply(map_score)
        return df
