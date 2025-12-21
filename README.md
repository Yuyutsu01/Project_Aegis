Project AEGIS

Multi-Model Agreement Trading Framework

Overview

Project AEGIS is an algorithmic trading research system that combines XGBoost, reinforcement learning (PPO), and market sentiment under a hard-consensus execution rule.

A trade is executed only when all models agree on direction.
If even one model disagrees, the system stays flat.

Example:

XGBoost → Long

PPO → Long

Sentiment → Short
→ No trade executed

This design reduces over-trading and false signals, making AEGIS suitable for risk-aware research and evaluation.

System Architecture

Market data is converted into technical features and fed into three independent decision modules:

XGBoost: Predicts next-day market direction from engineered features

PPO Agent: Learns trading actions from sequential market states

Sentiment Module: Confirms or vetoes trades using external sentiment data

All outputs flow into a hard-consensus engine.

Execution rule:

if xgb == ppo == sentiment:
    execute trade
else:
    stay flat


Portfolio updates and metrics are computed after execution, never during inference.

Execution Pipeline

Data Ingestion
Historical OHLC data is downloaded and stored as parquet files.

Feature Engineering
Generates indicators such as returns, RSI, MACD, volatility, and trend filters.

Example:
A strong upward trend with low volatility increases long-bias features.

XGBoost Training
Trains a directional classifier and saves the model artifact.

PPO Training
Trains a reinforcement learning agent in a custom trading environment.

Hard-Consensus Backtest
Executes trades only on full agreement and records equity, drawdown, and signals.

Visualization
Streamlit dashboard displays equity curve, drawdowns, positions, and signals.

Repository Structure
project_aegis/
├── data/                  # Raw and processed market data
├── src/
│   ├── envs/               # Trading environment (PPO)
│   ├── features/           # Feature engineering
│   └── consensus/          # Hard consensus logic
├── artifacts/
│   ├── xgb/                # Trained XGBoost models
│   ├── ppo/                # Trained PPO agents
│   └── backtests/          # Backtest results
├── run_* scripts           # Pipeline execution
├── streamlit_app.py
└── README.md

Performance Metrics

Evaluation focuses on risk-adjusted performance, not just returns:

Cumulative Return

Sharpe Ratio

Sortino Ratio

Calmar Ratio

Maximum Drawdown

Exposure Ratio

Example:
Lower returns with a Sharpe of 1.5 are preferred over higher returns with a Sharpe of 0.3.

Key Design Choices

Hard consensus instead of weighted averaging

No data leakage (training, backtesting, visualization separated)

Fixed transaction costs

Single-asset backtesting

No leverage

These constraints keep results interpretable and reproducible.

Intended Use

Academic research

Hybrid ML/RL experimentation

Risk-aware strategy design

Portfolio or system design demonstration

Not intended for live trading without further validation.

Future Extensions

Multi-asset trading

Regime-aware consensus

Dynamic position sizing

Walk-forward validation

Live market data integration

License

For research and educational use only.