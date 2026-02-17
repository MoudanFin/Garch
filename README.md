# MASI GARCH + Rolling Backtest + VaR

This repository contains a Python script to:

1. Download MASI index prices (Yahoo Finance ticker).
2. Compute MASI log returns from 2020 to 2025.
3. Select a compatible GARCH model on the training sample (by AIC).
4. Run a rolling-window one-step-ahead forecast on the test sample.
5. Compare forecasts with realized values.
6. Estimate and backtest Value-at-Risk (VaR 95% and 99%).

## Script

- `masi_garch_var.py`

## Install

```bash
pip install pandas numpy scipy yfinance arch
```

## Run

```bash
python masi_garch_var.py --ticker ^MASI --start 2020-01-01 --end 2025-12-31 --split-ratio 0.8 --window-size 750
```

If `^MASI` has no data in your environment, try another Yahoo Finance symbol such as `MASI.CS`.

## Outputs

Saved in `outputs/`:

- `masi_returns.csv`: return series.
- `rolling_backtest.csv`: rolling forecasts, realized returns, VaR, and breach flags.
- `summary_metrics.csv`: model choice and performance metrics.
