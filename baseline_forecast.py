"""
Milestone 1, Step 3: Baseline forecasts (seasonal naive).

Reads:   data/processed/load_clean.csv
Writes:  data/processed/baseline_forecasts.csv

Two baselines, per region:
  naive_24h  : forecast = load at the same hour yesterday
  naive_168h : forecast = load at the same hour one week ago

Scored on a holdout year (2025). Rows where the actual value OR either
forecast is missing (e.g. just after a data gap) are dropped, so both
baselines are always scored on exactly the same hours.

Run from the project folder:  python baseline_forecast.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("data/processed/load_clean.csv")
OUT = Path("data/processed/baseline_forecasts.csv")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
LAGS = {"naive_24h": 24, "naive_168h": 168}


def mape(actual: pd.Series, pred: pd.Series) -> float:
    return float((np.abs(actual - pred) / actual).mean() * 100)


def rmse(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.sqrt(((actual - pred) ** 2).mean()))


def main() -> None:
    df = pd.read_csv(CLEAN, parse_dates=["timestamp_utc"])

    frames, rows = [], []
    for region, g in df.groupby("region"):
        s = g.set_index("timestamp_utc")["load_mwh_clean"].sort_index()
        frame = pd.DataFrame({"actual": s})
        for name, lag in LAGS.items():
            frame[name] = s.shift(lag)  # shift by hours: index is complete hourly

        test = frame[frame.index >= TEST_START].dropna()
        for name in LAGS:
            rows.append(
                {
                    "region": region,
                    "baseline": name,
                    "hours_scored": len(test),
                    "MAPE_%": round(mape(test["actual"], test[name]), 2),
                    "RMSE_MW": round(rmse(test["actual"], test[name]), 0),
                }
            )
        test = test.assign(region=region)
        frames.append(test.reset_index())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)

    results = pd.DataFrame(rows).set_index(["region", "baseline"])
    print(f"Holdout: {TEST_START.date()} onward\n")
    print(results.to_string())
    print(f"\nForecasts saved to {OUT}")


if __name__ == "__main__":
    main()