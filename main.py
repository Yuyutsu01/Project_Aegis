import time
import schedule
from datetime import datetime

from config.settings import TRADING, MODELS, PORTFOLIO
from config.fyers_keys import FYERS_CREDENTIALS
from data.fyers_client import FyersDataClient
from data.news_client import NewsClient
from features.technical_indicators import TechnicalIndicators
from features.sentiment_analyzer import SentimentAnalyzer
from features.feature_engineer import FeatureEngineer
from models.supervised.xgboost_model import XGBoostModel
from models.reinforcement.ppo_agent import PPOTradingAgent
from trading.consensus_engine import ConsensusEngine
from trading.risk_manager import RiskManager
from trading.portfolio_manager import PortfolioManager
from trading.execution_engine import ExecutionEngine
from utils.logger import setup_logger

class AutonomousTradingSystem:
    def __init__(self):
        self.logger = setup_logger()
        self.setup_components()
        self.is_running = False
        
    def setup_components(self):
        """Initialize all system components"""
        # Data clients
        self.fyers_client = FyersDataClient()
        self.news_client = NewsClient()
        
        # Feature engineering
        self.technical_calc = TechnicalIndicators()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        
        # AI Models
        self.ml_model = XGBoostModel(MODELS.XGBOOST_MODEL_PATH)
        self.rl_agent = PPOTradingAgent(MODELS.RL_MODEL_PATH)
        
        # Trading components
        self.consensus_engine = ConsensusEngine(TRADING.CONSENSUS_THRESHOLD)
        self.risk_manager = RiskManager(TRADING.INITIAL_CAPITAL, TRADING.MAX_DRAWDOWN)
        self.portfolio_manager = PortfolioManager(
            TRADING.INITIAL_CAPITAL,
            PORTFOLIO.MAX_POSITIONS,
            PORTFOLIO.MAX_POSITION_SIZE
        )
        self.execution_engine = ExecutionEngine(self.fyers_client, self.portfolio_manager)
        
        self.logger.info(f"Autonomous Trading System initialized for {len(TRADING.SYMBOLS)} stocks")
    
    def enable_test_mode(self):
        """Temporarily lower thresholds to see more trading activity"""
        self.consensus_engine.threshold = 0.5  # Lower from 0.7 to 0.5
        self.logger.info("TEST MODE: Lowered consensus threshold to 0.5")
        self.logger.info("TEST MODE: This will generate more trading signals for demonstration")
    
    def get_stock_data(self, symbol: str):
        """Get complete data for a stock"""
        try:
            market_data = self.fyers_client.get_live_quote(symbol)
            historical_data = self.fyers_client.get_historical_data(symbol, TRADING.TIMEFRAME, 30)
            news_data = self.news_client.get_latest_news()
            
            if historical_data.empty:
                return None
            
            features = self.feature_engineer.prepare_features(historical_data, news_data)
            technical_signal = self.technical_calc.get_technical_signal(historical_data)
            
            return {
                'symbol': symbol,
                'market_data': market_data,
                'historical_data': historical_data,
                'features': features,
                'technical_signal': technical_signal,
                'current_price': market_data.get('current_price', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def evaluate_all_stocks(self):
        """Evaluate all stocks and return trading opportunities"""
        opportunities = []
        
        for symbol in TRADING.SYMBOLS:
            try:
                stock_data = self.get_stock_data(symbol)
                if stock_data is None:
                    continue
                
                # Get predictions from AI models
                ml_prediction, ml_confidence = self.ml_model.predict(stock_data['features'])
                rl_action = self.rl_agent.predict(stock_data['features'])  # Using features for RL too
                
                # Evaluate consensus
                evaluation = self.consensus_engine.evaluate_stock(
                    symbol, ml_prediction, ml_confidence, rl_action,
                    stock_data['features'].get('sentiment_score', 0),
                    stock_data['technical_signal']
                )
                
                opportunities.append(evaluation)
                
                # More detailed logging for debugging
                if evaluation['consensus']:
                    self.logger.info(f"STRONG BUY: {symbol} - Confidence: {evaluation['confidence']:.2f}")
                else:
                    self.logger.info(f"Evaluated {symbol}: Consensus={evaluation['consensus']}, Confidence={evaluation['confidence']:.2f}, Reasons: {', '.join(evaluation['reasons'])}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {symbol}: {e}")
        
        return opportunities
    
    def execute_multi_stock_trading(self):
        """Execute trading cycle for all stocks"""
        self.logger.info("=== Starting Multi-Stock Trading Cycle ===")
        
        # Evaluate all stocks
        opportunities = self.evaluate_all_stocks()
        buy_opportunities = [opp for opp in opportunities if opp['consensus']]
        
        self.logger.info(f"Found {len(buy_opportunities)} buy opportunities out of {len(opportunities)} stocks")
        
        # Sort by confidence and execute best opportunities
        buy_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        executed_trades = 0
        for opportunity in buy_opportunities:
            if executed_trades >= PORTFOLIO.MAX_POSITIONS:
                self.logger.info("Max positions reached, skipping further trades")
                break
                
            if not self.portfolio_manager.can_open_new_position(opportunity['symbol'], 0):
                self.logger.info(f"Cannot open new position for {opportunity['symbol']} - portfolio constraints")
                continue
            
            # Calculate position size
            available_capital = self.portfolio_manager.get_available_capital_for_trade()
            position_size = self.risk_manager.calculate_kelly_position(opportunity['confidence'])
            investment_amount = available_capital * position_size
            
            # Risk management check
            if self.risk_manager.check_risk_limits() and investment_amount > 1000:  # Minimum 1000 INR
                # Execute trade
                success = self.execution_engine.execute_order(
                    opportunity['symbol'],
                    investment_amount,
                    'BUY',
                    f"Multi-AI consensus: Confidence={opportunity['confidence']:.2f}"
                )
                
                if success:
                    executed_trades += 1
                    self.logger.info(f"SUCCESS: Opened position in {opportunity['symbol']} with INR {investment_amount:,.2f}")
                else:
                    self.logger.error(f"FAILED: Failed to open position in {opportunity['symbol']}")
            else:
                self.logger.warning(f"WARNING: Risk limits or insufficient amount for {opportunity['symbol']}")
        
        if executed_trades == 0 and buy_opportunities:
            self.logger.info("No trades executed due to risk limits or portfolio constraints")
        elif executed_trades == 0:
            self.logger.info("No consensus buy signals found")
        else:
            self.logger.info(f"Successfully executed {executed_trades} trades this cycle")
    
    def portfolio_health_check(self):
        """Check portfolio health"""
        current_prices = {}
        for symbol in TRADING.SYMBOLS:
            data = self.get_stock_data(symbol)
            if data:
                current_prices[symbol] = data['current_price']
        
        portfolio_stats = self.portfolio_manager.get_portfolio_stats(current_prices)
        self.logger.info(f"Portfolio Health: Value=INR {portfolio_stats['total_value']:,.2f}, "
                        f"PnL=INR {portfolio_stats['pnl']:,.2f} ({portfolio_stats['pnl_percent']:.1f}%), "
                        f"Positions={portfolio_stats['positions_count']}, "
                        f"Drawdown={portfolio_stats['drawdown']:.2%}")
    
    def run_autonomous(self):
        """Main autonomous trading loop"""
        self.is_running = True
        self.logger.info(f"Starting Autonomous Trading System for {len(TRADING.SYMBOLS)} stocks")
        
        # Schedule trading cycles
        schedule.every(TRADING.UPDATE_FREQUENCY).seconds.do(self.execute_multi_stock_trading)
        
        # Schedule portfolio health check
        schedule.every(5).minutes.do(self.portfolio_health_check)
        
        # Initial health check
        self.portfolio_health_check()
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down Autonomous Trading System...")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"System error: {e}")
            self.is_running = False

def main():
    """Main entry point"""
    trading_system = AutonomousTradingSystem()
    
    # Enable test mode for more trading activity
    trading_system.enable_test_mode()
    
    trading_system.run_autonomous()

if __name__ == "__main__":
    main()