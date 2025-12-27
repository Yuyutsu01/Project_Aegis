
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

        