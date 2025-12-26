import numpy as np

class RiskManager:
    """
    Publication-grade risk filters.
    These gates sit AFTER signal generation
    and BEFORE execution.
    """

    def __init__(
        self,
        max_volatility=0.03,
        max_drawdown=0.25,
        cooldown_period=5
    ):
        self.max_volatility = max_volatility
        self.max_drawdown = max_drawdown
        self.cooldown_period = cooldown_period

        self.equity_curve = []
        self.cooldown = 0

    # --------------------------------------------------
    # UPDATE EQUITY
    # --------------------------------------------------
    def update_equity(self, pnl: float):
        if len(self.equity_curve) == 0:
            self.equity_curve.append(pnl)
        else:
            self.equity_curve.append(self.equity_curve[-1] + pnl)

    # --------------------------------------------------
    # CURRENT DRAWDOWN
    # --------------------------------------------------
    def current_drawdown(self) -> float:
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        return np.min(eq - peak)

    # --------------------------------------------------
    # VOLATILITY GATE
    # --------------------------------------------------
    def volatility_gate(self, recent_returns: np.ndarray) -> bool:
        if len(recent_returns) < 20:
            return True  # not enough data → allow
        vol = np.std(recent_returns[-20:])
        return vol <= self.max_volatility

    # --------------------------------------------------
    # DRAWDOWN GATE
    # --------------------------------------------------
    def drawdown_gate(self) -> bool:
        if len(self.equity_curve) < 30:
            return True
        return abs(self.current_drawdown()) <= self.max_drawdown

    # --------------------------------------------------
    # COOLDOWN LOGIC
    # --------------------------------------------------
    def cooldown_gate(self) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        return True

    def register_trade_result(self, pnl: float):
        if pnl < 0:
            self.cooldown = self.cooldown_period
