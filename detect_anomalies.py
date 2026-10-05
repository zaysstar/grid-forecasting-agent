"""
Milestone 2b: Residual inspection and anomaly detection (v3).

Reads:   data/processed/model_forecasts.csv   (2025 out-of-sample forecasts)
         data/raw/weather_hourly.csv          (for temperature context)
Writes:  data/processed/anomaly_flags.csv     (every hour, with scores + flags)
         data/processed/anomaly_events.csv    (flagged hours grouped into events)

Rules
  spike rule      : |robust z of the hourly residual| > Z_THRESH
  sustained rule  : |robust z of the trailing ROLL_H-hour mean residual| > Z_THRESH
  flag            : either rule fires
Scores are scaled within each calendar month so seasons where the model is
normally less accurate are not over-flagged.

New in v3
  * Threshold sweep: flag rate and injection recall at several thresholds, so
    the sensitivity setting is a documented trade-off, not a guess.
  * Injection results store the peak z reached, so any threshold can be
    evaluated afterwards.

Reading the results
  Detection sensitivity is limited by model error: a miss must be roughly
  Z_THRESH times the typical error to be flagged. A more accurate model makes
  the detector more sensitive. On real data the sustained rule may add little
  because forecast errors persist for hours (averaging does not reduce them).

Run from the project folder:  python detect_anomalies.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/model_forecasts.csv")
WEATHER_PATH = Path("data/raw/weather_hourly.csv")
OUT_FLAGS = Path("data/processed/anomaly_flags.csv")
OUT_EVENTS = Path("data/processed/anomaly_events.csv")

Z_THRESH = 4.0                    # operating threshold
SWEEP = [3.0, 3.5, 4.0, 5.0]      # thresholds compared in the sweep table
ROLL_H = 6                        # window for the sustained-deviation rule
MERGE_GAP_H = 2                   # flagged hours this close form one event
TEMP_BASE_H = 720                 # temperature compared with the prior 30 days
TEMP_DEV_C = 7.0                  # |deviation| at the peak that counts as unusual
INJECT_MAGS = [0.05, 0.10, 0.20]  # injected size as a fraction of the forecast
N_PER_MAG = 200                   # injected events per size, per region. Each is
                                  # scored on its own, so they may overlap in time.
                                  # 200 per size gives about +/-7 points of
                                  # sampling error on a recall figure.
SEED = 42


def monthly_stats(s: pd.Series, month: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Per-row median and robust sigma (1.4826 * MAD), computed within each month."""
    med = s.groupby(month).transform("median")
    sigma = (s - med).abs().groupby(month).transform("median") * 1.4826
    return med, sigma.where(sigma > 0)


def trailing_mean(resid: pd.Series) -> pd.Series:
    return resid.rolling(f"{ROLL_H}h", min_periods=ROLL_H).mean()


def score(resid: pd.Series, stats: dict) -> tuple[pd.Series, pd.Series]:
    """Robust z-scores using FIXED statistics from the clean data."""
    z_spike = (resid - stats["med1"]) / stats["sig1"]
    z_sustained = (trailing_mean(resid) - stats["medr"]) / stats["sigr"]
    return z_spike, z_sustained


def temperature_context(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.set_index("timestamp_utc").sort_index()
    base = w["temp_c"].rolling(TEMP_BASE_H, min_periods=TEMP_BASE_H // 2).mean()
    return pd.DataFrame({"temp_c": w["temp_c"], "temp_dev_c": w["temp_c"] - base})


def weather_label(dev: float) -> str:
    if pd.isna(dev):
        return "unknown"
    if dev <= -TEMP_DEV_C:
        return "much colder than prior 30 days"
    if dev >= TEMP_DEV_C:
        return "much hotter than prior 30 days"
    return "no unusual temperature"


def find_events(g: pd.DataFrame) -> pd.DataFrame:
    flagged = g[g["flag"]]
    if flagged.empty:
        return pd.DataFrame()
    gap = flagged.index.to_series().diff() > pd.Timedelta(hours=MERGE_GAP_H)
    event_id = gap.cumsum().to_numpy()
    rows = []
    for _, grp in flagged.groupby(event_id):
        start, end = grp.index.min(), grp.index.max()
        peak = grp["z_score"].abs().idxmax()
        span = g.loc[start:end]
        spike_any, sus_any = grp["flag_spike"].any(), grp["flag_sustained"].any()
        rows.append(
            {
                "start_utc": start,
                "end_utc": end,
                "duration_h": int((end - start) / pd.Timedelta(hours=1)) + 1,
                "flagged_hours": len(grp),
                "trigger": "both" if spike_any and sus_any
                else ("single-hour spike" if spike_any else "sustained shift"),
                "peak_time_utc": peak,
                "peak_z": round(float(grp.loc[peak, "z_score"]), 1),
                "peak_actual_MW": round(float(grp.loc[peak, "actual"])),
                "peak_forecast_MW": round(float(grp.loc[peak, "pred_model"])),
                "mean_residual_MW": round(float(span["residual"].mean())),
                "direction": "above forecast" if grp.loc[peak, "z_score"] > 0 else "below forecast",
                "temp_c_at_peak": round(float(g.loc[peak, "temp_c"]), 1)
                if pd.notna(g.loc[peak, "temp_c"]) else np.nan,
                "temp_dev_c": round(float(g.loc[peak, "temp_dev_c"]), 1)
                if pd.notna(g.loc[peak, "temp_dev_c"]) else np.nan,
                "weather_context": weather_label(g.loc[peak, "temp_dev_c"]),
            }
        )
    return pd.DataFrame(rows)


def injection_test(g: pd.DataFrame, stats: dict, rng) -> pd.DataFrame:
    """Add known spikes/dips to the residuals and record the peak z each reaches.
    Scoring statistics stay fixed at what the clean data produced."""
    resid = g["residual"]
    pred = g["pred_model"].to_numpy()
    already = g["flag"].to_numpy()

    n = len(g)
    rows = []
    for mag in INJECT_MAGS:
        for _ in range(N_PER_MAG):
            # Draw every random value first, so the stream (and therefore the
            # injected set) does not change when the flag pattern changes.
            dur = int(rng.integers(3, 13))
            start = int(rng.integers(0, n - dur - ROLL_H))
            sign = rng.choice([-1, 1])
            end = start + dur
            check_end = end + ROLL_H - 1  # trailing mean flags late
            if already[start:check_end].any():
                continue  # a real anomaly is already here; would inflate recall
            rows.append(_score_injection(resid, pred, stats, mag, sign, start, end, check_end))
    return pd.DataFrame(rows)


def _score_injection(resid, pred, stats, mag, sign, start, end, check_end) -> dict:
    """Shift the residuals in [start, end) by sign * mag * forecast and return
    the peak |z| reached by each rule in the window (plus trailing hours)."""
    inj = resid.copy()
    inj.iloc[start:end] = inj.iloc[start:end].to_numpy() + sign * mag * pred[start:end]
    z_spike, z_sus = score(inj, stats)
    window = slice(start, check_end)
    return {
        "magnitude": mag,
        "max_z_spike": float(z_spike.iloc[window].abs().max()),
        "max_z_sus": float(z_sus.iloc[window].abs().max()),
    }


def hit_flags(inj: pd.DataFrame, rule: str, thr: float) -> pd.Series:
    if rule == "spike":
        z = inj["max_z_spike"]
    else:
        z = inj[["max_z_spike", "max_z_sus"]].max(axis=1)
    return z > thr


def recall_table(inj_all: pd.DataFrame, rule: str, thr: float) -> pd.DataFrame:
    tmp = inj_all.assign(hit=hit_flags(inj_all, rule, thr))
    table = tmp.pivot_table(
        index="region", columns="magnitude", values="hit", aggfunc="mean"
    ).mul(100).round(0)
    table.columns = [f"{int(c * 100)}%_size" for c in table.columns]
    return table.join(inj_all.groupby("region").size().rename("n_injected"))


def main() -> None:
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp_utc"])
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp_utc"])
    rng = np.random.default_rng(SEED)

    flag_frames, event_frames, inject_frames, summary = [], [], [], []
    sweep_rates = {}
    for region, g in df.groupby("region"):
        g = g.set_index("timestamp_utc").sort_index()
        month = g.index.month.to_numpy()

        # Fixed per-month statistics from the clean residuals
        resid = g["residual"]
        med1, sig1 = monthly_stats(resid, month)
        medr, sigr = monthly_stats(trailing_mean(resid), month)
        stats = {"med1": med1, "sig1": sig1, "medr": medr, "sigr": sigr}

        z_spike, z_sus = score(resid, stats)
        g["z_spike"], g["z_sustained"] = z_spike, z_sus
        g["flag_spike"] = z_spike.abs() > Z_THRESH
        g["flag_sustained"] = z_sus.abs() > Z_THRESH
        g["flag"] = g["flag_spike"] | g["flag_sustained"]
        sus_filled = z_sus.fillna(0)
        g["z_score"] = z_spike.where(z_spike.abs() >= sus_filled.abs(), z_sus)

        for t in SWEEP:
            sweep_rates[(region, t)] = float(
                ((z_spike.abs() > t) | (z_sus.abs() > t)).mean() * 100
            )

        ctx = temperature_context(weather[weather.region == region].drop(columns="region"))
        g = g.join(ctx, how="left")

        events = find_events(g)
        if not events.empty:
            events.insert(0, "region", region)
            event_frames.append(events)

        inj = injection_test(g, stats, rng)
        inj["region"] = region
        inject_frames.append(inj)

        overall_sigma = 1.4826 * (resid - resid.median()).abs().median()
        summary.append(
            {
                "region": region,
                "hours": len(g),
                "bias_MW": round(float(resid.mean())),
                "typical_error_MW": round(float(overall_sigma)),
                "spike_hrs": int(g["flag_spike"].sum()),
                "sustained_hrs": int(g["flag_sustained"].sum()),
                "flagged_hours": int(g["flag"].sum()),
                "flag_rate_%": round(float(g["flag"].mean() * 100), 2),
                "events": len(events),
            }
        )
        out = g.reset_index()[
            ["timestamp_utc", "actual", "pred_model", "residual", "z_spike",
             "z_sustained", "z_score", "flag_spike", "flag_sustained", "flag",
             "temp_c", "temp_dev_c"]
        ]
        out.insert(0, "region", region)
        flag_frames.append(out)

        flagged = g[g["flag"]]
        by_month = flagged.groupby(flagged.index.month).size()
        print(f"{region} flagged hours by month: "
              f"{ {int(m): int(n) for m, n in by_month.items()} }")

    OUT_FLAGS.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(flag_frames, ignore_index=True).to_csv(OUT_FLAGS, index=False)
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    events_all.to_csv(OUT_EVENTS, index=False)

    print("\n=== Residual inspection and flag rates (2025) ===")
    print(pd.DataFrame(summary).set_index("region").to_string())
    print(
        "\nflag_rate_% is the share of hours flagged on the REAL data with no "
        "injection. It is an upper bound on the false-positive rate, because "
        "some flagged hours may be genuine events."
    )

    print("\n=== Largest events per region (by peak |z|) ===")
    show = ["start_utc", "duration_h", "trigger", "peak_z", "mean_residual_MW",
            "direction", "temp_dev_c", "weather_context"]
    if not events_all.empty:
        for region, ev in events_all.groupby("region"):
            top = ev.reindex(ev["peak_z"].abs().sort_values(ascending=False).index).head(5)
            print(f"\n{region}")
            print(top[show].to_string(index=False, col_space=12))

    inj_all = pd.concat(inject_frames, ignore_index=True)
    print(f"\n=== Injection test at threshold {Z_THRESH}: share of synthetic anomalies caught ===")
    print("\nSingle-hour rule only:")
    print(recall_table(inj_all, "spike", Z_THRESH).to_string())
    print("\nSingle-hour + sustained rule:")
    print(recall_table(inj_all, "combined", Z_THRESH).to_string())
    print(
        "\nInjected anomalies last 3-12 hours and shift load up or down by the "
        "stated % of the forecast. A catch means at least one hour in or just "
        "after the window was flagged."
    )

    print("\n=== Threshold sweep (both rules): flag rate vs recall ===")
    rows = []
    for (region, t), rate in sweep_rates.items():
        sub = inj_all[inj_all.region == region]
        hit = hit_flags(sub, "combined", t)
        row = {"region": region, "threshold": t, "flag_rate_%": round(rate, 2)}
        for mag in INJECT_MAGS:
            row[f"recall_{int(mag * 100)}%_size"] = round(
                float(hit[sub["magnitude"] == mag].mean() * 100)
            )
        rows.append(row)
    print(pd.DataFrame(rows).set_index(["region", "threshold"]).to_string())
    print(
        "\nLower thresholds catch more injected anomalies but also flag more "
        "ordinary hours. Pick the operating point from this table."
    )
    print(f"\nSaved {OUT_FLAGS} and {OUT_EVENTS}")


if __name__ == "__main__":
    main()