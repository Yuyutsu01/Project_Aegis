import pandas as pd
from typing import Dict, List
from .fyers_client import FyersDataClient
from .yfinance_client import YFinanceClient
from .news_client import NewsClient
from config.settings import DATA, TRADING

class DataLoader:
    def __init__(self):
        self.fyers_client = FyersDataClient()
        self.yfinance_client = YFinanceClient()
        self.news_client = NewsClient()
    
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        try:
            return self.fyers_client.get_historical_data(symbol, TRADING.TIMEFRAME, days)
        except Exception as e:
            print(f"FYERS failed: {e}, using Yahoo Finance")
            return self.yfinance_client.get_historical_data(symbol, days)
    
    def get_live_data(self, symbol: str) -> Dict:
        return self.fyers_client.get_live_quote(symbol)
    
    def get_news_data(self) -> List[str]:
        return self.news_client.get_latest_news()
    
    def get_complete_dataset(self, symbol: str) -> Dict:
        return {
            'market_data': self.get_live_data(symbol),
            'historical_data': self.get_historical_data(symbol, DATA.HISTORICAL_DAYS),
            'news_data': self.get_news_data()
        }