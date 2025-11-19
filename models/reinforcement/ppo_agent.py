import numpy as np
import random

class PPOTradingAgent:
    def __init__(self, model_path: str = None):
        self.model = None
        self.learning_phase = True
        print("PPO Agent initialized (enhanced mock)")
    
    def predict(self, state) -> int:
        """Enhanced RL prediction with learning behavior"""
        if isinstance(state, dict):
            # More sophisticated rule-based logic
            rsi = state.get('rsi', 50)
            sentiment = state.get('sentiment_score', 0)
            price_vs_sma = state.get('price_vs_sma20', 1)
            momentum = state.get('momentum', 0)
            
            buy_signals = 0
            if rsi < 35: buy_signals += 2  # Strong buy
            elif rsi < 45: buy_signals += 1
            if sentiment > 0.3: buy_signals += 2  # Strong sentiment
            elif sentiment > 0.1: buy_signals += 1
            if price_vs_sma > 1.03: buy_signals += 1
            if momentum > 0.02: buy_signals += 1
            
            sell_signals = 0
            if rsi > 65: sell_signals += 2  # Strong sell
            elif rsi > 55: sell_signals += 1
            if sentiment < -0.3: sell_signals += 2
            elif sentiment < -0.1: sell_signals += 1
            if price_vs_sma < 0.97: sell_signals += 1
            if momentum < -0.02: sell_signals += 1
            
            # Add some exploration randomness
            if random.random() < 0.1:  # 10% exploration
                return random.choice([-1, 0, 1])
            
            if buy_signals >= 3:
                return 1  # Buy
            elif sell_signals >= 3:
                return -1  # Sell
            else:
                return 0  # Hold
        else:
            return random.choice([-1, 0, 1])  # More dynamic