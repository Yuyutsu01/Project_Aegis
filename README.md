# PROJECT AEGIS

**Meta-Policy Reinforcement Learning Framework for Risk-Aware Quantitative Trading**

---

## Executive Summary

**Project AEGIS** (Advanced Execution & Gradient-boosted Intelligent Strategy) is an enterprise-grade quantitative trading framework designed to solve the challenge of non-stationary environments and low signal-to-noise ratios in financial time series. Standard reinforcement learning agents struggle when trained directly on raw technical indicators, leading to training instability and high turnover. AEGIS addresses this by decoupling sensory sentiment extraction from execution and risk management logic.

A highly regularized **XGBoost Classifier** acts as the system's **Sensory Input (The Eyes)**, translating raw technical features into robust directional probabilities. A **Proximal Policy Optimization (PPO)** agent acts as the **Decision Maker (The Brain)**, learning to overlay risk management, inventory state, and transaction costs onto the classifier's signals. This combined design optimizes for long-term risk-adjusted returns (Sharpe/Sortino) and execution efficiency rather than just next-candle directional accuracy.

---

## Project Highlights

* **Decoupled Architecture:** Separates pattern recognition (XGBoost) from tactical trade execution (PPO).
* **Friction-Aware Environment:** Custom Gymnasium environment modeling 0.1% transaction costs (broker commission + slippage) to prevent overtrading.
* **Rolling Z-Scores:** Normalizes raw returns, moving averages, and volumes into rolling statistical z-scores to eliminate look-ahead bias and handle non-stationarity.
* **Production-Grade Configs:** Strict configuration separation (development, staging, production) validated using Pydantic schema models.
* **DevOps Infrastructure:** Multi-stage Docker deployment, pre-commit quality gates, and automated GitHub Actions testing pipelines.

---

## Monorepo Directory Target Structure

Project AEGIS is organized as an enterprise-grade monorepo:

```text
project-aegis/
├── apps/                               # Deployable user applications
│   ├── dashboard/                      # Streamlit-based Cyber-Quant Terminal
│   │   └── dashboard_app.py
│   └── terminal_api/                   # FastAPI gateway serving metrics and reports
│       └── api_app.py
├── packages/                           # Internal reusable packages
│   ├── core_utils/                     # Shared logger, error classes, and configs
│   ├── data_ingestor/                  # Ingestors for Yahoo Finance & Fyers broker
│   ├── feature_engineer/               # Rolling z-score calculators
│   └── trading_environment/            # Custom Gymnasium Trading Environment
├── src/                                # Model definitions, training, and backtesting
│   ├── models/
│   │   ├── ppo/
│   │   │   ├── agent.py
│   │   │   └── train.py
│   │   └── xgb/
│   │       ├── model.py
│   │       └── train.py
│   └── backtesting/
│       ├── engine.py
│       └── run_backtest.py
├── deployment/                         # DevOps and deployment configs
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/
├── configs/                            # Environment-specific configuration files
│   ├── base.yaml
│   ├── dev.yaml
│   └── prod.yaml
├── docs/                               # Developer and architecture manuals
├── knowledge/                          # Domain knowledge base
├── tests/                              # Testing suite (unit, integration, e2e)
├── monitoring/                         # Prometheus & Grafana configurations
└── research/                           # Notebooks and research reports
```

---

## Quick Start Guide

### Option 1: Multi-Service Docker Compose Setup (Recommended)
Make sure Docker Desktop is running, then execute:
```bash
cd deployment/docker/
docker compose up --build
```
This builds and launches the API Gateway (port `8000`), the Cyber-Quant Terminal Dashboard (port `8501`), and Prometheus monitoring (port `9090`).

### Option 2: Local Python Setup (Poetry)
1. **Install Poetry:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. **Install Project Dependencies:**
   ```bash
   poetry install
   ```
3. **Initialize Git Hooks:**
   ```bash
   poetry run pre-commit install
   ```
4. **Execute local pipeline steps:**
   ```bash
   # Run ingestion
   poetry run python scripts/data/run_ingestion.py
   # Run feature engineering
   poetry run python scripts/features/run_feature_engineer.py
   # Train models
   poetry run python src/models/xgb/train.py
   poetry run python src/models/ppo/train.py
   # Run backtest
   poetry run python src/backtesting/run_backtest.py
   ```

---

## Quantitative Research & Backtesting Benchmarks

Out-of-sample holdout metrics (evaluated under a 0.1% transaction cost assumption):

| Metric | Buy & Hold | XGBoost Only | PPO Only | AEGIS (Hybrid) |
| :--- | :--- | :--- | :--- | :--- |
| **Annualized Return** | 11.93% | 2.26% | -4.66% | **19.80%** |
| **Annualized Volatility** | 19.62% | 19.66% | 17.47% | **18.91%** |
| **Sharpe Ratio** | 0.70 | 0.17 | -0.19 | **1.08** |
| **Sortino Ratio** | 1.23 | 0.31 | -0.27 | **1.64** |
| **Max Drawdown** | -15.14% | -26.51% | -19.43% | **-12.97%** |
| **Profit Factor** | 1.13 | 1.03 | 0.97 | **1.21** |
| **Calmar Ratio** | 0.79 | 0.09 | -0.24 | **1.53** |

For details on indicator formulations, reward engineering, and walk-forward training cycles, please refer to the files in the [docs/](file:///c:/Users/shiva/OneDrive/Desktop/projects/Project_Aegis/docs) and [knowledge/](file:///c:/Users/shiva/OneDrive/Desktop/projects/Project_Aegis/knowledge) directories.

---

## Contributing

We welcome contributions to Project AEGIS. Please read our [Developer & Contribution Guide](file:///c:/Users/shiva/OneDrive/Desktop/projects/Project_Aegis/docs/DeveloperGuide.md) and [Contributing Guidelines](file:///c:/Users/shiva/OneDrive/Desktop/projects/Project_Aegis/docs/Contributing.md) for details on code style, linting standards, and pull request workflows.

---

## License & Disclaimer

This project is licensed under the MIT License. See [LICENSE](file:///c:/Users/shiva/OneDrive/Desktop/projects/Project_Aegis/LICENSE) for details.

*Disclaimer: This software is for educational and research purposes only. Quantitative trading involves significant risk of capital loss. Do not use this software for live trading without professional validation.*
