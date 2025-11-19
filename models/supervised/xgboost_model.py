import numpy as np
from typing import Tuple
import random

class XGBoostModel:
    def __init__(self, model_path: str = None):
        self.model = None
        print("XGBoost model initialized (enhanced mock)")
    
    def predict(self, features: dict) -> Tuple[int, float]:
        """Enhanced mock prediction with more realistic behavior"""
        if not features:
            return 0, 0.5
            
        # More sophisticated mock logic
        base_score = 0.5
        
        # RSI-based signals
        rsi = features.get('rsi', 50)
        if rsi < 30:
            base_score += 0.3  # Strong buy signal when oversold
        elif rsi < 40:
            base_score += 0.15
        elif rsi > 70:
            base_score -= 0.3  # Strong sell when overbought
        elif rsi > 60:
            base_score -= 0.15
            
        # Price vs SMA signals
        price_vs_sma = features.get('price_vs_sma20', 1)
        if price_vs_sma > 1.05:
            base_score += 0.2  # Strong bullish above SMA
        elif price_vs_sma > 1.02:
            base_score += 0.1
        elif price_vs_sma < 0.95:
            base_score -= 0.2  # Bearish below SMA
        elif price_vs_sma < 0.98:
            base_score -= 0.1
            
        # Sentiment influence
        sentiment = features.get('sentiment_score', 0)
        base_score += sentiment * 0.2
        
        # Volume confirmation
        volume_trend = features.get('volume_trend', 1)
        if volume_trend > 1.2:
            base_score += 0.1  # High volume confirmation
        
        # Add some randomness to simulate market noise
        base_score += random.uniform(-0.1, 0.1)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, base_score))
        prediction = 1 if score > 0.6 else (0 if score > 0.4 else -1)
        
        return prediction, score