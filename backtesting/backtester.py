import pandas as pd
from typing import Dict

class Backtester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
    
    def run_backtest(self, historical_data: pd.DataFrame) -> Dict:
        """Simple mock backtest"""
        return {
            'trades': [],
            'portfolio_values': [self.initial_capital] * len(historical_data) if not historical_data.empty else [self.initial_capital],
            'final_value': self.initial_capital * 1.05,  # Mock 5% gain
            'total_trades': 0
        }