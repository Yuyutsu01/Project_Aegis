# System Architecture Specification

This document details the architectural layers and interaction boundaries of Project AEGIS.

## Decoupled Meta-Policy Design

Standard reinforcement learning approaches feed raw, noisy market features directly to an agent. This often results in training instability and overfitting. AEGIS solves this using a decoupled hybrid approach:

```
+-----------------------------------+
|      Feature Engineer             |
|   (Transforms Raw data to Zs)     |
+-----------------------------------+
                  |
                  |  (Continuous Z-Scores)
                  v
+-----------------------------------+
|     XGBoost Classifier (Eyes)     |
|   (Predicts directional prob)     |
+-----------------------------------+
                  |
                  |  (Calibrated P_Up)
                  v
+-----------------------------------+
|       PPO Agent (Brain)           |
|   (Determines exposure policy)    |
+-----------------------------------+
```

### Components

1. **XGBoost Classifier (Sensory Level):** 
   - Receives continuous z-scores.
   - Outputs a scalar $P(\text{Up}) \in [0, 1]$.
   - Restrained to a maximum depth of 3 with high L2 regularization to filter out market noise.

2. **PPO Agent (Cognitive Policy Level):**
   - Receives $P(\text{Up})$ alongside current holding position and contextual indicators.
   - Learns a discrete decision policy over actions: Short ($-1$), Flat ($0$), Long ($1$).
   - Learns to hold or cut exposure when prediction confidence is low or transaction costs are high.

3. **OpenAI Gym Simulation Environment:**
   - Incorporates commissions and slippage.
   - Implements delayed rewards to prevent look-ahead bias (actions taken at close $t$ earn return at close $t+1$).
