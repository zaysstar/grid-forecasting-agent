"""
Milestone 1, Step 2: Clean the raw EIA load data.

Reads:   data/raw/eia_hourly_load.csv
Writes:  data/processed/load_clean.csv

What it does, per region:
  1. Rebuilds a complete hourly timeline (absent hours become explicit NaN rows).
  2. Flags physically implausible values as data errors (not real events).
  3. Interpolates ONLY short gaps (<= MAX_GAP_HOURS). Longer gaps stay NaN.
  4. Keeps the original value and a flag column for every repaired row,
     so nothing is silently changed.

Run from the project folder:  python clean_load.py
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw/eia_hourly_load.csv")
OUT = Path("data/processed/load_clean.csv")

# A value is treated as a data error if it falls outside this band around the
# region's median load. Wide on purpose so real heat waves and cold snaps
# survive. Check the printed summary to confirm real extremes are not clipped.
LOW_FRAC = 0.4
HIGH_FRAC = 2.0

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

    # 2. Flag implausible values
    median = raw.median()
    low, high = LOW_FRAC * median, HIGH_FRAC * median
    outlier = (raw < low) | (raw > high)
    masked = raw.mask(outlier)

    # 3. Interpolate short gaps only (whole gap or nothing)
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
        imputed=("flag_imputed", "sum"),
        still_missing=("flag_still_missing", "sum"),
        band_low=("band_low", "first"),
        band_high=("band_high", "first"),
        kept_min=("load_mwh_clean", "min"),
        kept_max=("load_mwh_clean", "max"),
    )
    print(summary.round(0).to_string())
    print(
        "\nCheck: kept_min / kept_max should sit inside the band and look like "
        "real extremes. For ERCO, CISO and DUK, outliers should be ~0."
    )


if __name__ == "__main__":
    main()