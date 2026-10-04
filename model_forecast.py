"""
Milestone 2, Step 2: Gradient boosting forecast with weather features.

Reads:   data/processed/load_clean.csv
         data/raw/weather_hourly.csv
Writes:  data/processed/model_forecasts.csv

One model per region. Trained on 2023-2024, tested on 2025, and scored on the
same hours as the baseline so the comparison is fair.

Setup:  pip install scikit-learn
Run:    python model_forecast.py

Day-ahead framing: the model only uses information that would be known a day
ahead (load from 24h and 168h earlier, calendar features, and weather at the
target hour). IMPORTANT LIMITATION: it uses OBSERVED weather as a stand-in for
a weather forecast, which makes results look somewhat better than a real
deployment would. State this in the README.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor

CLEAN = Path("data/processed/load_clean.csv")
WEATHER = Path("data/raw/weather_hourly.csv")
OUT = Path("data/processed/model_forecasts.csv")

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
TIMEZONES = {
    "ERCO": "America/Chicago",
    "CISO": "America/Los_Angeles",
    "DUK": "America/New_York",
    "FPL": "America/New_York",
}
FEATURES = [
    "hour", "dow", "month", "is_holiday",
    "temp_c", "humidity_pct", "temp_24h_mean",
    "lag_24h", "lag_168h",
]
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2022-12-01", end="2026-01-31")


def build_features(region: str, load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = (
        load.merge(weather, on="timestamp_utc", how="left")
        .set_index("timestamp_utc")
        .sort_index()
    )

    # Calendar features in LOCAL time (people follow the local clock, incl. DST)
    local = df.index.tz_convert(TIMEZONES[region])
    df["hour"] = local.hour
    df["dow"] = local.dayofweek
    df["month"] = local.month
    df["is_holiday"] = local.normalize().tz_localize(None).isin(HOLIDAYS).astype(int)

    # Lags (the index is a complete hourly timeline, so shift = hours)
    load_series = df["load_mwh_clean"]
    df["lag_24h"] = load_series.shift(24)
    df["lag_168h"] = load_series.shift(168)
    df["temp_24h_mean"] = df["temp_c"].rolling(24, min_periods=12).mean()
    return df


def mape(actual, pred) -> float:
    return float((np.abs(actual - pred) / actual).mean() * 100)


def rmse(actual, pred) -> float:
    return float(np.sqrt(((actual - pred) ** 2).mean()))


def main() -> None:
    load = pd.read_csv(CLEAN, parse_dates=["timestamp_utc"])
    weather = pd.read_csv(WEATHER, parse_dates=["timestamp_utc"])

    rows, outputs = [], []
    for region, g in load.groupby("region"):
        w = weather[weather.region == region].drop(columns="region")
        df = build_features(region, g.drop(columns="region"), w)
        df = df[df["load_mwh_clean"].notna()]

        train = df[df.index < TEST_START]
        test = df[df.index >= TEST_START].dropna(subset=["lag_24h", "lag_168h"])

        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0
        )
        model.fit(train[FEATURES], train["load_mwh_clean"])
        pred = pd.Series(model.predict(test[FEATURES]), index=test.index)

        actual = test["load_mwh_clean"]
        base = test["lag_24h"]  # same-hour-yesterday baseline
        m_mape, b_mape = mape(actual, pred), mape(actual, base)
        rows.append(
            {
                "region": region,
                "hours_scored": len(test),
                "model_MAPE_%": round(m_mape, 2),
                "baseline_MAPE_%": round(b_mape, 2),
                "MAPE_improvement_%": round((1 - m_mape / b_mape) * 100, 1),
                "model_RMSE_MW": round(rmse(actual, pred), 0),
                "baseline_RMSE_MW": round(rmse(actual, base), 0),
            }
        )
        outputs.append(
            pd.DataFrame(
                {
                    "region": region,
                    "actual": actual,
                    "pred_model": pred,
                    "pred_naive_24h": base,
                    "residual": actual - pred,
                }
            ).reset_index()
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(OUT, index=False)

    print(f"Holdout: {TEST_START.date()} onward\n")
    print(pd.DataFrame(rows).set_index("region").to_string())
    print(f"\nForecasts saved to {OUT}")


if __name__ == "__main__":
    main()