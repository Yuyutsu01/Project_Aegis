from typing import Dict, List
import numpy as np

class PerformanceAnalyzer:
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        if not returns or np.std(returns) == 0:
            return 0
        excess_returns = [r - risk_free_rate/252 for r in returns]
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: List[float]) -> float:
        if not portfolio_values:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def generate_performance_report(self, backtest_results: Dict, initial_capital: float) -> Dict:
        portfolio_values = backtest_results.get('portfolio_values', [initial_capital])
        
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        sharpe = self.calculate_sharpe_ratio(returns)
        max_dd = self.calculate_max_drawdown(portfolio_values)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': backtest_results.get('total_trades', 0)
        }