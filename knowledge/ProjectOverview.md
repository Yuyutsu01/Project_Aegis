# Project Overview & Business Problem

This document provides a high-level summary of Project AEGIS, detailing the core business problem and strategy mechanics.

## The Business Problem

Quantitative trading strategies that use deep learning and reinforcement learning (RL) directly on raw market data often perform poorly in production. The key issues are:

1. **High Noise-to-Signal Ratio:** Financial price charts are filled with random noise. Training large neural networks on raw data causes them to overfit, learning patterns that do not generalize out-of-sample.
2. **Execution Frictions (Slippage and Fees):** An agent trained without transaction cost modeling will trade too frequently (churning), losing all its profits to broker fees and slippage.
3. **Regime Shifts:** Markets are non-stationary, shifting between bull, bear, and range-bound conditions. Static models cannot adapt to these shifts, leading to large drawdowns.

## The AEGIS Solution

AEGIS (Advanced Execution & Gradient-boosted Intelligent Strategy) uses a decoupled, hybrid architecture to address these challenges:

- **XGBoost (Sensory Level):** A highly regularized, shallow decision-tree model that filters raw market data and predicts short-term directional trends.
- **PPO Reinforcement Learning (Cognitive Level):** An agent that learns a trading policy based on the predictions from the XGBoost model. It handles exposure management, sizing, and transaction cost optimization.
- **Transaction Cost Penalty:** A 0.1% transaction cost is modeled in the Gym environment step, training the PPO agent to keep positions open unless a high-conviction signal is received.
