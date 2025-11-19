import pandas as pd
import numpy as np
from typing import Dict

class YFinanceClient:
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Mock Yahoo Finance data"""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        data = {
            'Date': dates,
            'Open': np.random.normal(1000, 50, len(dates)),
            'High': np.random.normal(1010, 50, len(dates)),
            'Low': np.random.normal(990, 50, len(dates)),
            'Close': np.random.normal(1000, 50, len(dates)),
            'Volume': np.random.randint(1000, 10000, len(dates))
        }
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    def get_current_price(self, symbol: str) -> float:
        """Mock current price"""
        return 1000 + hash(symbol) % 500