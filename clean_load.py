"""
Milestone 1, Step 2: Clean the raw EIA load data.

Reads:   data/raw/eia_hourly_load.csv
Writes:  data/processed/load_clean.csv

What it does, per region:
  1. Rebuilds a complete hourly timeline (absent hours become explicit NaN rows).
  2. Rule 1, BAND CHECK: flags values far outside the region's normal range.
     Catches huge spikes and near-zero collapses.
  3. Rule 2, SPIKE CHECK: flags values that jump away from their own
     neighbours (compared with the median of the surrounding hours).
     Catches bad readings that sit inside the normal range, such as a one-hour
     dip to 11,800 MW in the middle of a 31,000 MW summer afternoon.
  4. Rule 3, ONE-HOUR SPIKE CHECK: flags an hour that is a local peak or
     trough AND more than POINT_SPIKE_FRAC away from the midpoint of its two
     neighbours. Catches a spike that lands inside a steep ramp, where the
     7-hour median in Rule 2 moves with the ramp and can miss it.
  5. Interpolates ONLY short gaps (<= MAX_GAP_HOURS). Longer gaps stay NaN.
  6. Keeps the original value and a flag column for every repaired row,
     so nothing is silently changed.

Run from the project folder:  python clean_load.py
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/eia_hourly_load.csv")
OUT = Path("data/processed/load_clean.csv")

# Rule 1: band around the region's median load. Wide on purpose.
LOW_FRAC = 0.4
HIGH_FRAC = 2.0

# Rule 2: an hour is a spike if it differs from the median of the surrounding
# SPIKE_WINDOW_H hours (centred, including itself) by more than SPIKE_FRAC.
# A median ignores a few bad hours, and on a steady rise or fall it equals the
# hour itself, so real ramps are not flagged. Real hour-to-hour moves are far
# smaller than 25%.
SPIKE_WINDOW_H = 7
SPIKE_FRAC = 0.25

# Rule 3: a local peak or trough this far from the midpoint of its two
# neighbours is a one-hour spike. On a steady ramp an hour sits BETWEEN its
# neighbours (not a peak or trough), and a real daily peak is within a few
# percent of the midpoint, so neither is flagged.
POINT_SPIKE_FRAC = 0.20

# Gaps up to this many hours get linear interpolation. Longer gaps stay NaN.
MAX_GAP_HOURS = 6


def clean_region(g: pd.DataFrame) -> pd.DataFrame:
    s = g.set_index("timestamp_utc")["load_mwh"].sort_index()

    # 1. Complete hourly timeline
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full_idx)
    s.index.name = "timestamp_utc"
    raw = s.copy()
    missing_raw = raw.isna()  # blank values AND hours absent from the file

    # 2. Rule 1: band check
    median = raw.median()
    low, high = LOW_FRAC * median, HIGH_FRAC * median
    band_outlier = (raw < low) | (raw > high)
    masked = raw.mask(band_outlier)

    # 3. Rule 2: local spike check (on the series with band outliers removed)
    local_median = masked.rolling(SPIKE_WINDOW_H, center=True, min_periods=4).median()
    spike = ((masked - local_median).abs() / local_median) > SPIKE_FRAC

    # 3b. Rule 3: one-hour spike check against the neighbours' midpoint
    prev_h, next_h = masked.shift(1), masked.shift(-1)
    midpoint = (prev_h + next_h) / 2
    local_extreme = ((masked - prev_h) * (masked - next_h)) > 0  # above or below BOTH
    point_spike = local_extreme & (
        ((masked - midpoint).abs() / midpoint) > POINT_SPIKE_FRAC
    )
    spike = spike | point_spike

    outlier = band_outlier | spike
    masked = raw.mask(outlier)

    # 4. Interpolate short gaps only (whole gap or nothing)
    gap = masked.isna()
    run_id = (gap != gap.shift()).cumsum()
    run_len = gap.groupby(run_id).transform("sum")
    short_gap = gap & (run_len <= MAX_GAP_HOURS)
    interpolated = masked.interpolate(method="time", limit_area="inside")
    clean = masked.where(~short_gap, interpolated)

    return pd.DataFrame(
        {
            "load_mwh_raw": raw,
            "load_mwh_clean": clean,
            "flag_missing_raw": missing_raw,
            "flag_outlier": outlier,
            "flag_spike": spike,  # subset of flag_outlier caught by Rule 2
            "flag_imputed": short_gap & clean.notna(),
            "flag_still_missing": clean.isna(),
            "band_low": low,
            "band_high": high,
        }
    )


def main() -> None:
    df = pd.read_csv(RAW, parse_dates=["timestamp_utc"])

    parts = []
    for region, g in df.groupby("region"):
        out = clean_region(g)
        out.insert(0, "region", region)
        parts.append(out.reset_index())
    result = pd.concat(parts, ignore_index=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns=["band_low", "band_high"]).to_csv(OUT, index=False)

    print(f"Saved {len(result):,} rows to {OUT}\n")
    summary = result.groupby("region").agg(
        hours=("load_mwh_clean", "size"),
        missing_raw=("flag_missing_raw", "sum"),
        outliers=("flag_outlier", "sum"),
        of_which_spikes=("flag_spike", "sum"),
        imputed=("flag_imputed", "sum"),
        still_missing=("flag_still_missing", "sum"),
        kept_min=("load_mwh_clean", "min"),
        kept_max=("load_mwh_clean", "max"),
    )
    print(summary.round(0).to_string())
    print(
        "\nCheck: of_which_spikes counts hours caught by the neighbour checks "
        "(Rules 2 and 3). Expect a small number: a few per region at most. A "
        "large count would mean a threshold is too tight and is flagging real "
        "swings. ERCO's peak (about 85,500 MW) should still be intact."
    )


if __name__ == "__main__":
    main()