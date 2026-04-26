<<<<<<< HEAD
# 🛡️ Project AEGIS
=======
# 🛡️ Project AEGIS 
>>>>>>> 08e3c73fccf7b4998ca817bf391a6bc9cad2c450
**RL-Optimized Meta-Policy Trading System**

[![Status](https://img.shields.io/badge/Status-Performance_Hardened-success?style=flat-square)](#-performance-hardening)
[![License](https://img.shields.io/badge/License-Research_Only-blue?style=flat-square)](#license)
[![Sharpe](https://img.shields.io/badge/OOS_Sharpe_Ratio-1.08-gold?style=flat-square)](#-performance-metrics)

Project **AEGIS** (Advanced Execution & Gradient-boosted Intelligent Strategy) is a hybrid quantitative trading system that leverages **Meta-Policy Reinforcement Learning**. Unlike traditional ensembles, AEGIS uses a high-performance XGBoost model as a primary signal generator ("The Eyes") and a PPO Reinforcement Learning agent as the decision maker ("The Brain").

---

## 💎 The Architecture: Meta-Policy Optimization
AEGIS moves away from brittle "Hard Consensus" to a dynamic **Meta-Policy Architecture**. 

| Component | Strategy Role | Logic Type |
| :--- | :--- | :--- |
| **XGBoost** | "The Eyes" | Predicts continuous market probabilities $P(\text{Up})$. |
| **PPO Agent** | "The Brain" | Optimizes trading actions based on XGB signals + market state. |

### How it Works:
1. **Feature Extraction:** Market technical indicators (RSI, MACD, Z-Scores) are calculated.
2. **Signal Generation:** A regularized XGBoost model predicts the directional probability of the next candle.
3. **RL Decision:** The PPO agent observes the XGBoost probability, the raw technicals, and its **current position**. It then decides whether to go Long, Short, or Flat, inherently optimizing for transaction costs and risk-adjusted returns.

---

## 📊 Performance Metrics
*Metrics verified via strict Out-of-Sample (OOS) chronological testing.*

| Metric | Value |
| :--- | :--- |
| **OOS Cumulative Return** | **+23.37%** |
| **OOS Sharpe Ratio** | **1.08** |
| **Max Drawdown** | **-19.38%** |
| **Exposure Ratio** | **93.20%** |
| **Transaction Cost** | **0.1% (Fixed)** |

---

<<<<<<< HEAD
## 🛠️ System Hardening (Audit Findings)
Following a rigorous quantitative audit of Version 1, AEGIS V2 has been hardened against:
1. **Overfitting:** Implemented L1/L2 regularization (`reg_alpha`, `reg_lambda`) and tree depth constraints in XGBoost.
2. **Logic Fragility:** Replaced the "Sentiment Veto" with a continuous Meta-Policy, removing the hidden long-bias that previously occurred when data was missing.
=======
## 🛠️ System Hardening
Following a rigorous quantitative audit, AEGIS has been hardened against:
1. **Overfitting:** Implemented L1/L2 regularization and tree depth constraints in XGBoost.
2. **Logic Fragility:** Replaced the "Sentiment Veto" with a continuous Meta-Policy, removing the hidden long-bias found in V1.
>>>>>>> 08e3c73fccf7b4998ca817bf391a6bc9cad2c450
3. **Execution Realism:** Integrated 0.1% commission directly into the RL reward function, forcing the agent to learn "sticky" positions that minimize churn.

---

## 🚀 Execution Flow

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution
To train and verify the system from scratch:
```bash
# Data Ingestion & Engineering
python run_ingestion.py
python run_feature_engineer.py

# Model Training
python run_xgb_training.py
python run_ppo_training.py

# Backtest & Verification
python run_meta_backtest.py
```

### 3. Dashboard Visualization
Launch the Cyber-Quant terminal to audit results:
```bash
streamlit run streamlit.py
```

---

## 📂 Repository Structure

```text
project_aegis/
├── src/
│   ├── envs/               # Meta-Policy Trading Environment
│   ├── models/             # XGBoost & PPO Agent Implementations
│   ├── data_ingestor.py    # Multi-source data ingestion
│   └── feature_engineer.py # Symbol-safe technical indicators
├── artifacts/              # Serialized Model Files & Results
├── data/                   # Processed Parquet Datasets
├── run_xgb_training.py     # Signal generator training
├── run_ppo_training.py     # Meta-policy training
├── run_meta_backtest.py    # Final V2 verification
└── streamlit.py            # V2 Cyber-Quant Dashboard
```

---

## ⚖️ License
For research and educational use only. This system is designed for backtesting experimentation. **Not intended for live trading without further validation.**
