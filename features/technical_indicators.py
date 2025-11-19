import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_sma(prices, window):
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices, window):
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def calculate_macd(prices):
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> dict:
        if data.empty:
            return {}
            
        closes = data['close'] if 'close' in data.columns else data['Close']
        volumes = data['volume'] if 'volume' in data.columns else data['Volume']
        
        return {
            'rsi': float(self.calculate_rsi(closes).iloc[-1]),
            'sma_20': float(self.calculate_sma(closes, 20).iloc[-1]),
            'sma_50': float(self.calculate_sma(closes, 50).iloc[-1]),
            'price_vs_sma20': float(closes.iloc[-1] / self.calculate_sma(closes, 20).iloc[-1]),
            'volume_trend': float(volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1),
            'momentum': float(closes.iloc[-1] / closes.iloc[-5] - 1) if len(closes) > 5 else 0.0
        }
    
    def get_technical_signal(self, data: pd.DataFrame) -> float:
        indicators = self.calculate_all_indicators(data)
        if not indicators:
            return 0.0
            
        signal = 0
        signal += 0.3 if indicators['rsi'] > 50 else -0.3
        signal += 0.3 if indicators['price_vs_sma20'] > 1 else -0.3
        signal += 0.2 if indicators['momentum'] > 0 else -0.2
        signal += 0.2 if indicators['volume_trend'] > 1 else -0.2
        
        return max(-1.0, min(1.0, signal))