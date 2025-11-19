from data.fyers_client import FyersDataClient
from .portfolio_manager import PortfolioManager
from data.database import LocalDatabase
from utils.logger import setup_logger

class ExecutionEngine:
    def __init__(self, fyers_client: FyersDataClient, portfolio_manager: PortfolioManager):
        self.fyers_client = fyers_client
        self.portfolio = portfolio_manager
        self.database = LocalDatabase()
        self.logger = setup_logger("ExecutionEngine")
    
    def execute_order(self, symbol: str, quantity: float, action: str, reason: str = "AI Consensus") -> bool:
        try:
            current_data = self.fyers_client.get_live_quote(symbol)
            current_price = current_data.get('current_price', 0)
            
            if current_price <= 0:
                self.logger.error(f"Invalid price for {symbol}")
                return False
            
            # Calculate share quantity based on amount
            shares = int(quantity / current_price)
            if shares == 0:
                self.logger.warning(f"Insufficient quantity for {symbol}")
                return False
            
            # Execute via FYERS
            success = self.fyers_client.place_order(symbol, shares, action)
            
            if success:
                self.portfolio.update_position(symbol, action, shares, current_price)
                self.database.save_trade(symbol, action, shares, current_price, reason)
                self.logger.info(f"Order executed: {action} {shares} of {symbol} at {current_price}")
                return True
            else:
                self.logger.error(f"Order failed for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return False