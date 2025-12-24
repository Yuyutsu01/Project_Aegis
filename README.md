# Project AEGIS  
**Agreement-Based Hybrid ML Trading System**

## Overview

Project AEGIS is a research-oriented algorithmic trading system that combines:

- Supervised learning (XGBoost)
- Reinforcement learning (PPO)
- Market sentiment signals

Trades are executed **only when all models agree on direction**.  
If there is disagreement, the system stays flat.

**Example:**

- XGBoost → Long  
- PPO → Long  
- Sentiment → Short  

→ **No trade executed**

This agreement-based approach reduces over-trading and focuses on higher-confidence setups.

---

## System Architecture

Market data is transformed into technical features and processed by three independent modules:

- **XGBoost**: Learns directional bias from engineered features  
- **PPO Agent**: Learns trading actions in a custom environment  
- **Sentiment Module**: Confirms or blocks trades  

A **consensus gate** checks alignment before execution.

```text
if xgb_signal == ppo_action == sentiment_signal:
    execute trade
else:
    stay flat

```

Portfolio updates and performance metrics are computed **after trade execution**, never during model inference.



## Execution Pipeline

### Data Ingestion
Historical OHLC data is downloaded and stored as parquet files.

### Feature Engineering
Generates technical indicators such as returns, RSI, MACD, volatility, and trend filters.

**Example:**  
A strong upward trend with low volatility increases long-bias features.

### XGBoost Training
Trains a directional classifier and saves the trained model artifact.

### PPO Training
Trains a reinforcement learning agent in a custom trading environment.

### Hard-Consensus Backtest
Executes trades only on full agreement and records equity, drawdown, and signals.

### Visualization
Streamlit dashboard displays the equity curve, drawdowns, positions, and signals.



## Repository Structure

```text
project_aegis/
├── data/                  # Raw and processed market data
├── src/
│   ├── envs/              # Trading environment (PPO)
│   ├── features/          # Feature engineering
│   └── consensus/         # Hard consensus logic
├── artifacts/
│   ├── xgb/               # Trained XGBoost models
│   ├── ppo/               # Trained PPO agents
│   └── backtests/         # Backtest results
├── run_* scripts          # Pipeline execution
├── streamlit_app.py
└── README.md

```

## Performance Metrics

Evaluation focuses on **risk-adjusted performance**, not just raw returns:

1. Cumulative Return  
2. Sharpe Ratio  
3. Sortino Ratio  
4. Calmar Ratio  
5. Maximum Drawdown  
6. Exposure Ratio  

**Example:**  
Lower returns with a Sharpe ratio of **1.5** are preferred over higher returns with a Sharpe of **0.3**.



## Key Design Choices

- Agreement-based execution instead of weighted averaging  
- No data leakage (training, backtesting, and visualization are separated)  
- Fixed transaction costs  
- Single-asset backtesting  
- No leverage  

These constraints keep results **interpretable and reproducible**.



## Intended Use

- Academic research.  
- Hybrid ML/RL experimentation.  
- Risk-aware strategy design. 
- Portfolio or system design demonstration.  

**Not intended for live trading without further validation.**



## Future Extensions

- Multi-asset trading  
- Regime-aware consensus  
- Dynamic position sizing  
- Walk-forward validation  
- Live market data integration  


## License

For research and educational use only.
