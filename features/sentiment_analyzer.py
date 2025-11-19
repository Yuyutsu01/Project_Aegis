import random
from typing import List

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_history = []
    
    def analyze_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis with market trends"""
        if not text:
            return random.uniform(-0.2, 0.2)
        
        # More varied sentiment based on content
        positive_words = ['positive', 'strong', 'growth', 'profit', 'gain', 'better', 'bullish', 'rally', 'surge']
        negative_words = ['negative', 'weak', 'loss', 'drop', 'worse', 'volatility', 'bearish', 'decline', 'fall']
        
        text_lower = text.lower()
        positive_score = sum(2 for word in positive_words if word in text_lower)  # Stronger weight
        negative_score = sum(2 for word in negative_words if word in text_lower)
        
        if positive_score + negative_score == 0:
            # More varied neutral sentiment
            return random.uniform(-0.3, 0.3)
        
        base_sentiment = (positive_score - negative_score) / (positive_score + negative_score)
        
        # Add some market noise
        base_sentiment += random.uniform(-0.1, 0.1)
        
        return max(-1.0, min(1.0, base_sentiment))
    
    def batch_analyze(self, texts: List[str]) -> List[float]:
        return [self.analyze_sentiment(text) for text in texts]