#!/usr/bin/env python3
"""MASI return modeling with GARCH, rolling forecast backtest, and VaR evaluation.

Workflow:
1. Download MASI index prices from Yahoo Finance.
2. Compute log returns from 2020 to 2025.
3. Select a compatible GARCH model on the training sample.
4. Run rolling-window 1-step-ahead forecasts on the test sample.
5. Compare forecasts with realized returns/volatility.
6. Compute and backtest VaR.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.stats import norm, t


@dataclass
class ModelSpec:
    p: int
    q: int
    dist: str
    mean: str
    aic: float


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(
            f"No data downloaded for ticker '{ticker}'. Try another ticker (e.g., MASI.CS or ^MASI)."
        )
    close = data["Close"].dropna()
    close.name = "close"
    return close


def compute_log_returns(close: pd.Series) -> pd.Series:
    returns = 100 * np.log(close / close.shift(1)).dropna()
    returns.name = "log_return_pct"
    return returns


def choose_best_garch(train: pd.Series, p_max: int = 2, q_max: int = 2) -> ModelSpec:
    candidates: List[ModelSpec] = []
    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            for dist in ("normal", "t"):
                for mean in ("Zero", "Constant"):
                    try:
                        model = arch_model(train, mean=mean, vol="GARCH", p=p, q=q, dist=dist)
                        fitted = model.fit(disp="off")
                        candidates.append(ModelSpec(p=p, q=q, dist=dist, mean=mean, aic=fitted.aic))
                    except Exception:
                        continue

    if not candidates:
        raise RuntimeError("Could not fit any GARCH specification on the training sample.")

    best = min(candidates, key=lambda x: x.aic)
    return best


def fit_model(series: pd.Series, spec: ModelSpec):
    model = arch_model(
        series,
        mean=spec.mean,
        vol="GARCH",
        p=spec.p,
        q=spec.q,
        dist=spec.dist,
    )
    return model.fit(disp="off")


def rolling_forecast(
    returns: pd.Series,
    split_idx: int,
    spec: ModelSpec,
    window_size: int,
) -> pd.DataFrame:
    test_dates = returns.index[split_idx:]
    if len(test_dates) == 0:
        raise ValueError("Test sample is empty. Reduce split ratio or provide more data.")

    rows: List[Dict] = []
    for i in range(split_idx, len(returns)):
        window_start = max(0, i - window_size)
        sample = returns.iloc[window_start:i]
        if len(sample) < max(spec.p, spec.q) + 20:
            continue

        res = fit_model(sample, spec)
        fc = res.forecast(horizon=1, reindex=False)

        mu_hat = float(fc.mean.iloc[-1, 0])
        var_hat = float(fc.variance.iloc[-1, 0])
        sigma_hat = math.sqrt(max(var_hat, 0.0))

        realized = float(returns.iloc[i])

        # VaR in return units (%).
        alpha_95 = 0.05
        alpha_99 = 0.01
        if spec.dist == "t":
            nu = float(res.params.get("nu", np.nan))
            # Standardized t quantile to match arch variance scaling.
            q95 = t.ppf(alpha_95, df=nu) * math.sqrt((nu - 2) / nu)
            q99 = t.ppf(alpha_99, df=nu) * math.sqrt((nu - 2) / nu)
        else:
            q95 = norm.ppf(alpha_95)
            q99 = norm.ppf(alpha_99)

        var95 = mu_hat + sigma_hat * q95
        var99 = mu_hat + sigma_hat * q99

        rows.append(
            {
                "date": returns.index[i],
                "realized_return": realized,
                "pred_mean": mu_hat,
                "pred_variance": var_hat,
                "pred_sigma": sigma_hat,
                "VaR_95": var95,
                "VaR_99": var99,
                "breach_95": int(realized < var95),
                "breach_99": int(realized < var99),
            }
        )

    return pd.DataFrame(rows).set_index("date")


def kupiec_test(breaches: pd.Series, alpha: float) -> Dict[str, float]:
    n = int(breaches.count())
    x = int(breaches.sum())
    if n == 0:
        return {"n": 0, "violations": 0, "exp_rate": alpha, "obs_rate": np.nan, "LR_uc": np.nan}

    p_hat = x / n
    # Avoid log(0)
    eps = 1e-12
    p_hat = min(max(p_hat, eps), 1 - eps)
    alpha = min(max(alpha, eps), 1 - eps)

    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr_uc = -2 * (ll_null - ll_alt)

    return {
        "n": n,
        "violations": x,
        "exp_rate": alpha,
        "obs_rate": x / n,
        "LR_uc": lr_uc,
    }


def evaluate(backtest_df: pd.DataFrame) -> Dict[str, float]:
    err_mean = backtest_df["realized_return"] - backtest_df["pred_mean"]
    mse_mean = float(np.mean(err_mean**2))
    rmse_mean = float(np.sqrt(mse_mean))

    realized_var_proxy = backtest_df["realized_return"] ** 2
    mse_var = float(np.mean((realized_var_proxy - backtest_df["pred_variance"]) ** 2))

    eps = 1e-12
    qlike = float(
        np.mean(
            np.log(backtest_df["pred_variance"] + eps)
            + realized_var_proxy / (backtest_df["pred_variance"] + eps)
        )
    )

    kupiec95 = kupiec_test(backtest_df["breach_95"], alpha=0.05)
    kupiec99 = kupiec_test(backtest_df["breach_99"], alpha=0.01)

    out = {
        "mean_rmse": rmse_mean,
        "mean_mse": mse_mean,
        "variance_mse": mse_var,
        "qlike": qlike,
        "var95_violations": kupiec95["violations"],
        "var95_expected_rate": kupiec95["exp_rate"],
        "var95_observed_rate": kupiec95["obs_rate"],
        "var95_lr_uc": kupiec95["LR_uc"],
        "var99_violations": kupiec99["violations"],
        "var99_expected_rate": kupiec99["exp_rate"],
        "var99_observed_rate": kupiec99["obs_rate"],
        "var99_lr_uc": kupiec99["LR_uc"],
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASI GARCH + rolling backtest + VaR")
    parser.add_argument("--ticker", default="^MASI", help="Yahoo Finance ticker for MASI")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train ratio in (0,1)")
    parser.add_argument(
        "--window-size",
        type=int,
        default=750,
        help="Rolling window size (number of observations)",
    )
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    close = download_prices(args.ticker, args.start, args.end)
    returns = compute_log_returns(close)

    split_idx = int(len(returns) * args.split_ratio)
    if split_idx <= 0 or split_idx >= len(returns):
        raise ValueError("Invalid split ratio for available data.")

    train = returns.iloc[:split_idx]

    spec = choose_best_garch(train)
    backtest_df = rolling_forecast(
        returns=returns,
        split_idx=split_idx,
        spec=spec,
        window_size=args.window_size,
    )
    metrics = evaluate(backtest_df)

    returns.to_csv(outdir / "masi_returns.csv", header=True)
    backtest_df.to_csv(outdir / "rolling_backtest.csv")

    summary = {
        "ticker": args.ticker,
        "start": args.start,
        "end": args.end,
        "n_obs": len(returns),
        "train_obs": len(train),
        "test_obs": len(backtest_df),
        "selected_model": f"GARCH({spec.p},{spec.q}) mean={spec.mean} dist={spec.dist}",
        "selected_aic": spec.aic,
    }
    summary.update(metrics)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)

    print("=== Selected model ===")
    print(summary["selected_model"])
    print(f"AIC: {summary['selected_aic']:.4f}")
    print("\n=== Backtest metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
