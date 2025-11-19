import os
from dataclasses import dataclass
from typing import List

@dataclass
class TradingConfig:
    SYMBOLS: List[str] = None
    TIMEFRAME: str = "5"
    INITIAL_CAPITAL: float = 100000.0
    MAX_DRAWDOWN: float = 0.02
    CONSENSUS_THRESHOLD: float = 0.7
    UPDATE_FREQUENCY: int = 60
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = [
                "NSE:RELIANCE-EQ",
                "NSE:TCS-EQ", 
                "NSE:INFY-EQ",
                "NSE:HDFCBANK-EQ",
                "NSE:ICICIBANK-EQ"
            ]

@dataclass
class ModelConfig:
    XGBOOST_MODEL_PATH: str = "models/supervised/xgboost_model.pkl"
    RL_MODEL_PATH: str = "models/reinforcement/ppo_model.zip"
    SENTIMENT_MODEL: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

@dataclass
class DataConfig:
    HISTORICAL_DAYS: int = 30
    DATA_SAVE_PATH: str = "data/local_storage/"

@dataclass  
class PortfolioConfig:
    MAX_POSITIONS: int = 3
    MAX_POSITION_SIZE: float = 0.3

# Global configurations
TRADING = TradingConfig()
MODELS = ModelConfig()
DATA = DataConfig()
PORTFOLIO = PortfolioConfig()