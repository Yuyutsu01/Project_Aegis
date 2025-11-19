class RiskManager:
    def __init__(self, initial_capital: float, max_drawdown: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
    
    def calculate_kelly_position(self, confidence: float) -> float:
        win_prob = confidence
        win_loss_ratio = 2.0
        kelly_fraction = win_prob - (1 - win_prob) / win_loss_ratio
        return max(0.0, min(0.5, kelly_fraction * 0.5))  # Conservative
    
    def check_risk_limits(self) -> bool:
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        return drawdown <= self.max_drawdown
    
    def update_capital(self, new_value: float):
        self.current_capital = new_value