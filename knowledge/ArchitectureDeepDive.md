# Architecture Deep Dive

This document provides a detailed breakdown of the components in the decoupled XGBoost + PPO architecture.

## Component Breakdown

```
Raw Daily Price Series (OHLCV)
             │
             ▼
+------------------------------------------+
| Feature Engineering (Rolling Z-Scores)   |
+------------------------------------------+
             │
             ▼
+------------------------------------------+
| XGBoost Classifier (Sensory Level)       |
| - Shallow trees (max_depth=3)            |
| - High regularization (reg_lambda=1.5)   |
| - Outputs probability P(Up)              |
+------------------------------------------+
             │
             ▼
+------------------------------------------+
| Custom Trading Gym Env (Execution Level)  |
| - Current Position (-1, 0, 1)            |
| - Transaction Cost (0.1%)                |
| - Outputs state: [Z-Scores, P_Up, Pos]   |
+------------------------------------------+
             │
             ▼
+------------------------------------------+
| PPO Actor-Critic Policy (Cognitive Level) |
| - Maps state to actions                  |
| - Maximizes friction-adjusted returns    |
+------------------------------------------+
```

### 1. Ingestion & Feature Normalization
- Downloads daily price data and flattens MultiIndex headers.
- Computes returns, simple moving averages, RSI, MACD, and volume ratios.
- Standardizes features using a rolling 60-day window z-score:
  $$feature_z = \frac{feature_t - \mu_{\text{rolling}, 60}}{\sigma_{\text{rolling}, 60}}$$
  This transforms non-stationary time series into rolling normalized distributions.

### 2. XGBoost Predictor (The Eyes)
- Trained on the first 60% of the dataset to predict if the next day's close is higher than the current day's close.
- Shallow trees (`max_depth=3`) and high regularization (`reg_alpha=0.5`, `reg_lambda=1.5`) are used to prevent the model from overfitting to noise.

### 3. PPO Meta-Policy (The Brain)
- Trained on the next 20% of the dataset (the validation split of the XGBoost model).
- The state space includes the 11 z-scores, the XGBoost probability $P(\text{Up})$, and the current holding position.
- Uses Stable-Baselines3 PPO with a standard Multi-Layer Perceptron (MLP) policy.
- Learns to trade selectively, entering positions only when the expected return outweighs the transaction cost of execution.
