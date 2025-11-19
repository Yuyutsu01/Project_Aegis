from typing import Dict, List

class ConsensusEngine:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate_stock(self, symbol: str, ml_prediction: int, ml_confidence: float, 
                      rl_action: int, sentiment_score: float, technical_signal: float) -> Dict:
        ml_agree = ml_confidence > self.threshold and ml_prediction == 1
        rl_agree = rl_action == 1
        sentiment_agree = sentiment_score > 0.1
        technical_agree = technical_signal > 0
        
        consensus_met = all([ml_agree, rl_agree, sentiment_agree, technical_agree])
        
        return {
            'symbol': symbol,
            'consensus': consensus_met,
            'confidence': ml_confidence,
            'ml_signal': ml_prediction,
            'rl_signal': rl_action,
            'sentiment': sentiment_score,
            'technical': technical_signal,
            'reasons': self._get_rejection_reasons(ml_agree, rl_agree, sentiment_agree, technical_agree) if not consensus_met else ["All systems agree"]
        }
    
    def _get_rejection_reasons(self, ml_agree: bool, rl_agree: bool, sentiment_agree: bool, technical_agree: bool) -> List[str]:
        reasons = []
        if not ml_agree: reasons.append("ML low confidence")
        if not rl_agree: reasons.append("RL not buying") 
        if not sentiment_agree: reasons.append("Negative sentiment")
        if not technical_agree: reasons.append("Technical bearish")
        return reasons