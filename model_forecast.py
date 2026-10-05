"""
Milestone 2, Step 2: Gradient boosting forecast with weather features.

Reads:   data/processed/load_clean.csv
         data/raw/weather_hourly.csv
Writes:  data/processed/model_forecasts.csv

Three candidate models per region:
  v1     predicts load in MW directly.
  v2     predicts load RELATIVE TO THE RECENT LEVEL (mean of the 7 days ending
         24h before the target hour), then scales back to MW. Helps when demand
         drifts upward, because trees cannot predict outside the range they saw
         in training.
  blend  the average of v1 and v2.

SELECTION WITHOUT PEEKING AT THE TEST YEAR
  1. Train each candidate on 2023 only, score it on 2024 (validation).
  2. Pick the best candidate per region by 2024 MAPE.
  3. Retrain on 2023-2024 and score 2025 (the test year) once.
All candidates are still reported on 2025 for transparency, but the choice was
made before the test year was looked at.

Setup:  pip install scikit-learn
Run:    python model_forecast.py

Day-ahead framing: only information known a day ahead is used (load at least
24h old, calendar features, weather at the target hour). LIMITATION: observed
weather stands in for a weather forecast, so results are somewhat optimistic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor

CLEAN = Path("data/processed/load_clean.csv")
WEATHER = Path("data/raw/weather_hourly.csv")
OUT = Path("data/processed/model_forecasts.csv")

PRIMARY = "auto"  # "auto" = choose by validation; or force "v1" / "v2" / "blend"
SELECT_START = pd.Timestamp("2024-01-01", tz="UTC")  # validation year begins
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
TIMEZONES = {
    "ERCO": "America/Chicago",
    "CISO": "America/Los_Angeles",
    "DUK": "America/New_York",
    "FPL": "America/New_York",
}
CALENDAR_WEATHER = [
    "hour", "dow_eff", "month", "is_holiday",
    "temp_c", "humidity_pct", "temp_24h_mean",
]
FEATURES_V1 = CALENDAR_WEATHER + ["lag_24h", "lag_168h"]
FEATURES_V2 = CALENDAR_WEATHER + ["lag_24h_rel", "lag_168h_rel"]
REQUIRED = ["lag_24h", "lag_168h", "level"]
_NAMED = USFederalHolidayCalendar().holidays(
    start="2022-12-01", end="2026-01-31", return_name=True
)
HOLIDAYS = _NAMED.index  # every federal holiday (observed dates)
# Major holidays behave like Sundays for load. Treating them as Sundays in the
# day-of-week feature lets the model reuse the weekend pattern it has seen
# hundreds of times, instead of learning each holiday from only two examples.
MAJOR_HOLIDAYS = _NAMED[
    _NAMED.str.contains("New Year|Memorial|Independence|Labor|Thanksgiving|Christmas")
    & ~_NAMED.str.contains("Juneteenth")  # its name contains "Independence"
].index


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
    local_date = local.normalize().tz_localize(None)
    df["is_holiday"] = local_date.isin(HOLIDAYS).astype(int)
    # effective day of week: major holidays count as Sunday (6)
    df["dow_eff"] = np.where(local_date.isin(MAJOR_HOLIDAYS), 6, df["dow"])

    # Lags (the index is a complete hourly timeline, so shift = hours)
    s = df["load_mwh_clean"]
    df["lag_24h"] = s.shift(24)
    df["lag_168h"] = s.shift(168)
    df["temp_24h_mean"] = df["temp_c"].rolling(24, min_periods=12).mean()

    # v2: recent level = mean load over the 7 days ending 24h before the target
    df["level"] = s.shift(24).rolling(168, min_periods=120).mean()
    df["lag_24h_rel"] = df["lag_24h"] / df["level"]
    df["lag_168h_rel"] = df["lag_168h"] / df["level"]
    return df


def mape(actual, pred) -> float:
    return float((np.abs(actual - pred) / actual).mean() * 100)


def rmse(actual, pred) -> float:
    return float(np.sqrt(((actual - pred) ** 2).mean()))


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0
    )


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train v1 and v2 on `train`, return forecasts for v1, v2 and their blend."""
    m1 = new_model().fit(train[FEATURES_V1], train["load_mwh_clean"])
    p1 = pd.Series(m1.predict(test[FEATURES_V1]), index=test.index)

    train2 = train.dropna(subset=["level"])
    m2 = new_model().fit(train2[FEATURES_V2], train2["load_mwh_clean"] / train2["level"])
    p2 = pd.Series(m2.predict(test[FEATURES_V2]) * test["level"].to_numpy(), index=test.index)

    return {"v1": p1, "v2": p2, "blend": (p1 + p2) / 2}


def main() -> None:
    load = pd.read_csv(CLEAN, parse_dates=["timestamp_utc"])
    weather = pd.read_csv(WEATHER, parse_dates=["timestamp_utc"])

    validation, accuracy, detail, outputs = [], [], [], []
    for region, g in load.groupby("region"):
        w = weather[weather.region == region].drop(columns="region")
        df = build_features(region, g.drop(columns="region"), w)
        df = df[df["load_mwh_clean"].notna()]

        # --- Selection: train on 2023, validate on 2024 ---
        train_sel = df[df.index < SELECT_START]
        valid = df[(df.index >= SELECT_START) & (df.index < TEST_START)].dropna(subset=REQUIRED)
        val_preds = fit_predict(train_sel, valid)
        val_mape = {k: mape(valid["load_mwh_clean"], p) for k, p in val_preds.items()}
        chosen = min(val_mape, key=val_mape.get) if PRIMARY == "auto" else PRIMARY
        validation.append(
            {"region": region, **{f"{k}_MAPE_%": round(v, 2) for k, v in val_mape.items()},
             "chosen": chosen}
        )

        # --- Final: train on 2023-2024, score the 2025 test year ---
        train = df[df.index < TEST_START]
        test = df[df.index >= TEST_START].dropna(subset=REQUIRED)
        preds = fit_predict(train, test)

        actual = test["load_mwh_clean"]
        base = test["lag_24h"]  # same-hour-yesterday baseline
        b_mape = mape(actual, base)
        pm = {k: mape(actual, p) for k, p in preds.items()}
        accuracy.append(
            {
                "region": region,
                "hours_scored": len(test),
                "baseline_MAPE_%": round(b_mape, 2),
                "v1_MAPE_%": round(pm["v1"], 2),
                "v2_MAPE_%": round(pm["v2"], 2),
                "blend_MAPE_%": round(pm["blend"], 2),
                "chosen": chosen,
                "chosen_vs_baseline_%": round((1 - pm[chosen] / b_mape) * 100, 1),
            }
        )
        primary = preds[chosen]
        detail.append(
            {
                "region": region,
                "baseline_RMSE_MW": round(rmse(actual, base)),
                "chosen_RMSE_MW": round(rmse(actual, primary)),
                "chosen_bias_MW": round(float((actual - primary).mean())),
            }
        )
        outputs.append(
            pd.DataFrame(
                {
                    "region": region,
                    "actual": actual,
                    "pred_model": primary,
                    "model_used": chosen,
                    "pred_v1": preds["v1"],
                    "pred_v2": preds["v2"],
                    "pred_blend": preds["blend"],
                    "pred_naive_24h": base,
                    "residual": actual - primary,
                }
            ).reset_index()
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(OUT, index=False)

    print("Selection step: trained on 2023, scored on 2024 (MAPE, lower is better)")
    print(pd.DataFrame(validation).set_index("region").to_string())
    print(f"\nTest year ({TEST_START.year}): trained on 2023-2024, MAPE")
    print(pd.DataFrame(accuracy).set_index("region").to_string())
    print("\nChosen model: error size and bias (bias = mean of actual - forecast;")
    print("positive means the model runs low)")
    print(pd.DataFrame(detail).set_index("region").to_string())
    print(f"\nForecasts saved to {OUT}")


if __name__ == "__main__":
    main()