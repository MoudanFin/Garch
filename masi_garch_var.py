"""MASI index return modeling with GARCH, rolling backtest, and VaR estimation.

Usage:
    python masi_garch_var.py --ticker MASI.CS --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
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
    vol: str
    p: int
    q: int
    o: int
    dist: str

    def to_label(self) -> str:
        return f"{self.vol}(p={self.p},o={self.o},q={self.q},dist={self.dist})"


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Try another Yahoo Finance symbol (for MASI, candidates include 'MASI.CS')."
        )

    close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[close_col].dropna().rename("price")
    if prices.empty:
        raise ValueError("Downloaded dataset does not contain valid close prices.")
    return prices


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute percentage log-returns for ARCH-family models."""
    returns = 100 * np.log(prices / prices.shift(1))
    return returns.dropna().rename("returns")


def candidate_specs() -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for dist in ["normal", "t"]:
        for p in [1, 2]:
            for q in [1, 2]:
                specs.append(ModelSpec(vol="GARCH", p=p, q=q, o=0, dist=dist))
                specs.append(ModelSpec(vol="EGARCH", p=p, q=q, o=0, dist=dist))
                specs.append(ModelSpec(vol="GARCH", p=p, q=q, o=1, dist=dist))
    return specs


def fit_spec(returns: pd.Series, spec: ModelSpec):
    am = arch_model(
        returns,
        mean="Constant",
        vol=spec.vol,
        p=spec.p,
        o=spec.o,
        q=spec.q,
        dist=spec.dist,
        rescale=False,
    )
    return am.fit(disp="off", show_warning=False)


def select_best_model(train_returns: pd.Series) -> Tuple[ModelSpec, object, pd.DataFrame]:
    """Grid-search model specs and pick the one with minimum AIC."""
    results: List[Dict[str, float]] = []
    best_spec: ModelSpec | None = None
    best_fit = None
    best_aic = np.inf

    for spec in candidate_specs():
        try:
            fitted = fit_spec(train_returns, spec)
            aic = fitted.aic
            bic = fitted.bic
            results.append({"model": spec.to_label(), "aic": aic, "bic": bic})
            if aic < best_aic:
                best_aic = aic
                best_spec = spec
                best_fit = fitted
        except Exception:
            continue

    if best_spec is None or best_fit is None:
        raise RuntimeError("No GARCH-family model could be fit on training returns.")

    ranking = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    return best_spec, best_fit, ranking


def one_step_rolling_forecast(
    returns: pd.Series,
    split_index: int,
    spec: ModelSpec,
) -> pd.DataFrame:
    """Re-fit model recursively and produce one-step-ahead forecasts on test sample."""
    predictions: List[Dict[str, float]] = []

    for i in range(split_index, len(returns)):
        train_slice = returns.iloc[:i]
        test_date = returns.index[i]
        realized = returns.iloc[i]

        fitted = fit_spec(train_slice, spec)
        fcast = fitted.forecast(horizon=1, reindex=False)

        mu_hat = float(fcast.mean.iloc[-1, 0])
        var_hat = float(fcast.variance.iloc[-1, 0])
        sigma_hat = float(np.sqrt(max(var_hat, 1e-12)))

        predictions.append(
            {
                "date": test_date,
                "realized_return": realized,
                "forecast_return": mu_hat,
                "forecast_variance": var_hat,
                "forecast_sigma": sigma_hat,
            }
        )

    pred_df = pd.DataFrame(predictions).set_index("date")
    pred_df["squared_error_return"] = (
        pred_df["realized_return"] - pred_df["forecast_return"]
    ) ** 2
    pred_df["realized_abs_return"] = pred_df["realized_return"].abs()
    pred_df["realized_sq_return"] = pred_df["realized_return"] ** 2
    return pred_df


def add_var_columns(pred_df: pd.DataFrame, fitted_model) -> pd.DataFrame:
    """Add 95% and 99% one-day VaR columns using the fitted distribution."""
    distribution = fitted_model.model.distribution.name.lower()
    params = fitted_model.params

    alpha_levels = [0.95, 0.99]
    out = pred_df.copy()

    if "student" in distribution:
        nu = float(params.get("nu", 8.0))
        for alpha in alpha_levels:
            q = t.ppf(1 - alpha, df=nu)
            out[f"VaR_{int(alpha*100)}"] = out["forecast_return"] + out["forecast_sigma"] * q
    else:
        for alpha in alpha_levels:
            q = norm.ppf(1 - alpha)
            out[f"VaR_{int(alpha*100)}"] = out["forecast_return"] + out["forecast_sigma"] * q

    for alpha in alpha_levels:
        key = f"VaR_{int(alpha*100)}"
        out[f"breach_{int(alpha*100)}"] = (out["realized_return"] < out[key]).astype(int)

    return out


def summarize_backtest(pred_df: pd.DataFrame) -> pd.DataFrame:
    rmse_return = float(np.sqrt(pred_df["squared_error_return"].mean()))
    mae_return = float(
        np.mean(np.abs(pred_df["realized_return"] - pred_df["forecast_return"]))
    )
    vol_proxy_corr = float(pred_df["forecast_sigma"].corr(pred_df["realized_abs_return"]))

    summary = {
        "rmse_return": rmse_return,
        "mae_return": mae_return,
        "corr_sigma_vs_abs_return": vol_proxy_corr,
        "var95_breach_rate": float(pred_df["breach_95"].mean()),
        "var99_breach_rate": float(pred_df["breach_99"].mean()),
    }
    return pd.DataFrame([summary])


def run_pipeline(
    ticker: str,
    start: str,
    end: str,
    split_ratio: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = download_prices(ticker=ticker, start=start, end=end)
    returns = compute_log_returns(prices)

    split_index = int(len(returns) * split_ratio)
    train_returns = returns.iloc[:split_index]

    best_spec, best_fit, ranking = select_best_model(train_returns)
    rolling = one_step_rolling_forecast(returns=returns, split_index=split_index, spec=best_spec)
    with_var = add_var_columns(rolling, best_fit)
    summary = summarize_backtest(with_var)

    prices.to_csv(output_dir / "prices.csv")
    returns.to_csv(output_dir / "returns.csv")
    ranking.to_csv(output_dir / "garch_model_ranking.csv", index=False)
    with_var.to_csv(output_dir / "rolling_forecast_backtest.csv")
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    print("=" * 80)
    print("MASI Return Modelling & Risk Report")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Period: {start} -> {end}")
    print(f"Observations: {len(returns)} | Train: {len(train_returns)} | Test: {len(with_var)}")
    print(f"Selected model (AIC): {best_spec.to_label()}")
    print()
    print("Top 5 model candidates by AIC:")
    print(ranking.head(5).to_string(index=False))
    print()
    print("Backtest summary:")
    print(summary.to_string(index=False))
    print()
    print(
        "Outputs written to:",
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASI GARCH + rolling forecast + VaR backtest")
    parser.add_argument("--ticker", default="MASI.CS", help="Yahoo Finance ticker for MASI index")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Fraction of sample used for training before rolling test begins",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write CSV outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        split_ratio=args.split_ratio,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
