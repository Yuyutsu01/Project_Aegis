# Feature Engineering Reference

This document describes the technical indicators and rolling normalization methods used in Project AEGIS.

## Indicator Specifications

The system calculates 11 primary features from raw daily Open, High, Low, Close, and Volume (OHLCV) price bars:

1. **`ret_1d`**: Daily percentage return.
   $$\text{ret\_1d}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$
2. **`ret_5d`**: 5-day percentage return.
3. **`sma_20` & `sma_50`**: Distance from the 20-day and 50-day Simple Moving Averages.
4. **`rsi`**: 14-day Relative Strength Index (RSI). Calculated numerically using rolling means of gains and losses, replacing zero values with a tiny epsilon (`1e-10`) to avoid division-by-zero errors.
5. **`macd` & `macd_signal`**: Moving Average Convergence Divergence line and signal line, using standard 12-day and 26-day spans.
6. **`bb_width`**: Bollinger Band width, calculated as:
   $$\text{bb\_width}_t = \frac{2 \cdot \sigma_{\text{rolling}, 20}(\text{Close})}{\mu_{\text{rolling}, 20}(\text{Close})}$$
7. **`vol_ratio`**: Daily volume divided by its 20-day moving average, highlighting volume spikes and liquidity changes.
8. **`trend_sma`**: A binary trend indicator that outputs $1$ if the 20-day SMA is above the 50-day SMA, and $0$ otherwise.

## Rolling Normalization (Z-Scores)

Time series data is non-stationary (means and standard deviations change over time). Standardizing features using a rolling 60-day window ensures they remain scale-invariant and stationary.

Continuous features (excluding the binary `trend_sma` feature) are normalized using:
$$\text{feature}_z = \frac{\text{feature}_t - \mu_{\text{rolling}, 60}}{\sigma_{\text{rolling}, 60}}$$
Where the standard deviation $\sigma$ is replaced by `1e-10` if it is equal to zero, avoiding division-by-zero errors.
