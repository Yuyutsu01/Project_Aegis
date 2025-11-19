from typing import List
import random

class NewsClient:
    def get_latest_news(self, query: str = "stock market") -> List[str]:
        """More realistic and varied news headlines"""
        news_pool = [
            "Market shows strong positive trends with earnings growth",
            "Economic indicators point to stable growth in the sector",
            "Company reports better than expected quarterly results",
            "Market volatility expected to decrease in coming sessions",
            "Sector analysis reveals new investment opportunities",
            "Global markets show mixed signals amid economic data",
            "Technical analysis suggests potential breakout patterns",
            "Investors show cautious optimism in current market conditions",
            "Trading volume increases as institutional investors participate",
            "Market sentiment turns positive after recent developments",
            "Some concerns about market overvaluation emerge",
            "Trading range narrows as market seeks direction",
            "Mixed signals from various sectors create uncertainty",
            "Short-term volatility expected due to external factors",
            "Analysts divided on near-term market direction"
        ]
        
        # Return 3-5 random news items with varied sentiment
        num_news = random.randint(3, 5)
        return random.sample(news_pool, num_news)