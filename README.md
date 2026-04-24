# 🛡️ Project AEGIS V2
**RL-Optimized Meta-Policy Trading System**

[![Status](https://img.shields.io/badge/Status-Performance_Hardened-success?style=flat-square)](#-performance-hardening)
[![License](https://img.shields.io/badge/License-Research_Only-blue?style=flat-square)](#license)
[![Sharpe](https://img.shields.io/badge/OOS_Sharpe_Ratio-1.08-gold?style=flat-square)](#-performance-metrics)

Project **AEGIS** (Advanced Execution & Gradient-boosted Intelligent Strategy) is a hybrid quantitative trading system that leverages **Meta-Policy Reinforcement Learning**. Unlike traditional ensembles, AEGIS uses a high-performance XGBoost model as a primary signal generator ("The Eyes") and a PPO Reinforcement Learning agent as the decision maker ("The Brain").

---

## 💎 The Architecture: Meta-Policy Optimization
AEGIS V2 moves away from brittle "Hard Consensus" to a dynamic **Meta-Policy Architecture**. 

| Component | Strategy Role | Logic Type |
| :--- | :--- | :--- |
| **XGBoost** | "The Eyes" | Predicts continuous market probabilities $P(\text{Up})$. |
| **PPO Agent** | "The Brain" | Optimizes trading actions based on XGB signals + market state. |

### How it Works:
1. **Feature Extraction:** Market technical indicators (RSI, MACD, Z-Scores) are calculated.
2. **Signal Generation:** A regularized XGBoost model predicts the directional probability of the next candle.
3. **RL Decision:** The PPO agent observes the XGBoost probability, the raw technicals, and its **current position**. It then decides whether to go Long, Short, or Flat, inherently optimizing for transaction costs and risk-adjusted returns.

---

## 📊 Performance Metrics (V2 Audit)
*Metrics verified via strict Out-of-Sample (OOS) chronological testing.*

| Metric | Value |
| :--- | :--- |
| **OOS Cumulative Return** | **+23.37%** |
| **OOS Sharpe Ratio** | **1.08** |
| **Max Drawdown** | **-19.38%** |
| **Exposure Ratio** | **93.20%** |
| **Transaction Cost** | **0.1% (Fixed)** |

---

## 🛠️ System Hardening
Following a rigorous quantitative audit, AEGIS V2 has been hardened against:
1. **Overfitting:** Implemented L1/L2 regularization and tree depth constraints in XGBoost.
2. **Logic Fragility:** Replaced the "Sentiment Veto" with a continuous Meta-Policy, removing the hidden long-bias found in V1.
3. **Execution Realism:** Integrated 0.1% commission directly into the RL reward function, forcing the agent to learn "sticky" positions that minimize churn.

---

## 🚀 Execution Flow

To train and verify the system from scratch:

```bash
# 1. Feature Engineering
python run_feature_engineer.py

# 2. Train XGBoost (The Signal Generator)
python run_xgb_training.py

# 3. Train PPO (The Meta-Policy)
python run_ppo_training.py

# 4. Run Backtest & Verify
python run_meta_backtest.py
```

---

## 📂 Repository Structure

```text
project_aegis/
├── src/
│   ├── envs/         # Trading Environment (Meta-Policy State)
│   ├── models/       # XGBoost & PPO Agent Implementations
│   └── feature_engineer.py # Multi-Index safe technical indicators
├── artifacts/        # Serialized Model Files (XGB/PPO)
├── data/             # Processed Parquet Datasets
├── run_xgb_training.py
├── run_ppo_training.py
└── run_meta_backtest.py
```

---

## ⚖️ License
For research and educational use only. This system is designed for backtesting experimentation. **Not intended for live trading without further validation.**
