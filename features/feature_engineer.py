import pandas as pd
from typing import Dict, List
from .technical_indicators import TechnicalIndicators
from .sentiment_analyzer import SentimentAnalyzer

class FeatureEngineer:
    def __init__(self):
        self.technical = TechnicalIndicators()
        self.sentiment = SentimentAnalyzer()
    
    def prepare_features(self, market_data: pd.DataFrame, news_data: List[str]) -> Dict:
        features = {}
        
        # Technical features
        if not market_data.empty:
            features.update(self.technical.calculate_all_indicators(market_data))
        
        # Sentiment features
        features.update(self._get_sentiment_features(news_data))
        
        return features
    
    def _get_sentiment_features(self, news_data: List[str]) -> Dict:
        if not news_data:
            return {'sentiment_score': 0.0, 'news_count': 0}
        
        sentiment_scores = self.sentiment.batch_analyze(news_data)
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        return {
            'sentiment_score': float(avg_sentiment),
            'news_count': len(news_data),
            'positive_news_ratio': sum(1 for s in sentiment_scores if s > 0.1) / len(sentiment_scores)
        }