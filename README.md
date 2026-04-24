# 🛡️ Project AEGIS
**Precision-Engineered Agreement-Based Hybrid ML Trading System**

[![Status](https://img.shields.io/badge/Status-Performance_Hardened-success?style=flat-square)](#-performance-hardening)
[![License](https://img.shields.io/badge/License-Research_Only-blue?style=flat-square)](#license)
[![Sharpe](https://img.shields.io/badge/Sharpe_Ratio-2.95-gold?style=flat-square)](#-live-performance-metrics)

Project **AEGIS** (Agreement-based Execution & Integrated Sentiment) is a high-conviction algorithmic trading framework designed to eliminate over-trading by enforcing strict consensus between multiple machine learning models.

---

## 💎 The Core Thesis: "Total Agreement"
Unlike systems that use weighted averages or "soft" ensembles, AEGIS operates on a **Hard Consensus Gate**. A trade is only executed if all underlying logic modules agree on the market direction.

| Signal Source | Strategy Role | Logic Type |
| :--- | :--- | :--- |
| **XGBoost** | Directional Bias | Supervised Classifer |
| **PPO Agent** | Action Policy | Reinforcement Learning |
| **Sentiment Engine** | Market Pulse | Veto/Confirmation |

> [!IMPORTANT]
> **Consensus Logic:** If `XGB` == `PPO` == `Sentiment`, the gate opens. Otherwise, the system remains **Flat (Neutral)**. This approach prioritizes alpha quality over trade frequency.

---

## 📊 Performance Metrics (Current Audit)
*Metrics extracted from the latest hardened backtest in `src/config.yaml`.*

| Metric | Value |
| :--- | :--- |
| **Cumulative Return** | **602.43%** |
| **Annualized Return** | **34.02%** |
| **Sharpe Ratio** | **2.954** |
| **Max Drawdown** | **-16.15%** |
| **Exposure Ratio** | **17.65%** |
| **Transaction Cost** | **0.1% (Fixed)** |

---

## 🛠️ Performance Hardening
The system has recently undergone a major **Logic Integrity Audit** to ensure professional-grade reliability:

1. **Zero Look-Ahead Bias**: Fixed reward calculation logic. Observations at time $t$ result in actions that earn returns at time $t+1$ (next candle).
2. **Realistic Friction**: Integrated commission and slippage modeling (set at 0.1% per trade) directly into the Environment reward signals.
3. **Transaction Ledger**: A new transparency layer that logs every signal-consensus event for forensic audit.

---

## 🖥️ Dashboard (Cyber-Quant UI)
Visualize live backtests and model agreement using our custom Streamlit interface:

*   **Alpha Curve**: Visualize equity growth vs. market benchmarks.
*   **Audit Tab**: Forensic ledger showing every trade, model signal, and PnL impact.
*   **Drawdown Profile**: Real-time risk monitoring.

```bash
# Launch the dashboard
streamlit run streamlit.py
```

---

## 📂 Repository Structure

```text
project_aegis/
├── src/
│   ├── envs/         # Hardened Trading Environment (PPO)
│   ├── features/     # Feature Engineering & Indicator Logic
│   └── consensus/    # The Hard-Consensus Gate Logic
├── artifacts/        # Serialized Model Files (XGB/PPO)
├── data/             # Market Datasets
└── streamlit.py      # Main Visualization Engine
```

---

## 🚀 Getting Started

1. **Configure**: Update `src/config.yaml` with your parameters.
2. **Train**: Run training scripts for XGBoost and PPO.
3. **Audit**: Run the consensus backtest to generate artifacts.
4. **Visualize**: Launch the Streamlit dashboard to audit results.

---

## ⚖️ License
For research and educational use only. This system is designed for backtesting experimentation. **Not intended for live trading without further validation.**

<<<<<<< HEAD
=======
---

>>>>>>> 60dd11352a428a3584fab14a14ed13066da64bc7
