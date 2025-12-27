import numpy as np

class HardConsensusEngine:
    """
    Publication-grade hard consensus engine.

    Trade ONLY if:
    - XGBoost and PPO agree directionally
    - Neither is flat
    - Confidence threshold is met
    """

    def __init__(self, xgb_margin=0.1):
        """
        xgb_margin:
            Minimum deviation from 0.5 probability
            required to consider XGB confident.
        """
        self.xgb_margin = xgb_margin

    # --------------------------------------------------
    # XGBOOST SIGNAL
    # --------------------------------------------------
    def xgb_signal(self, prob: float) -> int:
        """
        Convert XGBoost probability to directional signal.
        """
        if prob >= 0.5 + self.xgb_margin:
            return 1
        elif prob <= 0.5 - self.xgb_margin:
            return -1
        else:
            return 0

    # --------------------------------------------------
    # PPO SIGNAL
    # --------------------------------------------------
    def ppo_signal(self, action: int) -> int:
        """
        PPO action already directional.
        """
        if action == 2:
            return 1
        elif action == 0:
            return -1
        else:
            return 0

    # --------------------------------------------------
    # HARD CONSENSUS DECISION
    # --------------------------------------------------
    def decide(self, xgb_prob: float, ppo_action: int) -> int:
        """
        Final trade decision.
        Returns:
            -1 → short
             0 → flat
            +1 → long
        """
        sx = self.xgb_signal(xgb_prob)
        sp = self.ppo_signal(ppo_action)

        if sx == sp and sx != 0:
            return sx
        return 0


# src/consensus/hard_consensus.py

class HardConsensus:
    """
    Hard consensus decision engine.
    Supports:
    - XGB + PPO
    - XGB + PPO + Sentiment (veto-only)
    """

    @staticmethod
    def decide(xgb_signal: int, ppo_signal: int) -> int:
        """
        Original 2-model hard consensus.
        """
        if xgb_signal == ppo_signal and xgb_signal != 0:
            return xgb_signal
        return 0

    @staticmethod
    def decide_with_sentiment(
        xgb_signal: int,
        ppo_signal: int,
        sentiment_signal: int
    ) -> int:
        """
        3-model hard consensus with sentiment veto.
        Sentiment can ONLY block trades. (added safety)
        """
        if (
            xgb_signal == ppo_signal ==
            sentiment_signal and
            xgb_signal != 0
        ):
            return xgb_signal

        return 0
