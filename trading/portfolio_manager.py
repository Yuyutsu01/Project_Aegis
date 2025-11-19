from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime

class PortfolioManager:
    def __init__(self, initial_capital: float, max_positions: int = 3, max_position_size: float = 0.3):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.max_positions = max_positions
        self.max_position_size = max_position_size
    
    def can_open_new_position(self, symbol: str, proposed_amount: float) -> bool:
        if len(self.positions) >= self.max_positions:
            return False
        if proposed_amount > self.initial_capital * self.max_position_size:
            return False
        if symbol in self.positions:
            return False
        return True
    
    def get_available_capital_for_trade(self) -> float:
        return min(self.current_cash, self.initial_capital * self.max_position_size)
    
    def update_position(self, symbol: str, action: str, quantity: float, price: float):
        if action.upper() == 'BUY':
            cost = quantity * price
            if cost <= self.current_cash:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    total_quantity = pos.quantity + quantity
                    avg_price = ((pos.quantity * pos.entry_price) + (quantity * price)) / total_quantity
                    self.positions[symbol] = Position(symbol, total_quantity, avg_price, price, datetime.now())
                else:
                    self.positions[symbol] = Position(symbol, quantity, price, price, datetime.now())
                self.current_cash -= cost
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        total_value = self.current_cash
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.current_price)
            total_value += position.quantity * current_price
        return total_value
    
    def get_portfolio_stats(self, current_prices: Dict[str, float]) -> Dict:
        current_value = self.get_portfolio_value(current_prices)
        return {
            'total_value': current_value,
            'cash': self.current_cash,
            'pnl': current_value - self.initial_capital,
            'pnl_percent': (current_value - self.initial_capital) / self.initial_capital * 100,
            'positions_count': len(self.positions),
            'drawdown': max(0, (self.initial_capital - current_value) / self.initial_capital)
        }