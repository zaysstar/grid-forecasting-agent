"""
Milestone 1, Step 1: Pull hourly regional demand from the EIA Open Data API (v2).

Setup:
    pip install requests pandas
    export EIA_API_KEY="your_key_here"      # free key: https://www.eia.gov/opendata/

Run:
    python pull_eia_load.py

Output:
    data/raw/eia_hourly_load.csv  (columns: timestamp_utc, region, load_mwh)
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
REGIONS = ["ERCO", "CISO", "DUK", "FPL"]  # ERCOT, CAISO, Duke Carolinas, FPL
START = "2023-01-01T00"  # UTC, format YYYY-MM-DDTHH
END = "2025-12-31T23"
PAGE_SIZE = 5000  # EIA's max rows per request
OUT_PATH = Path("data/raw/eia_hourly_load.csv")


def fetch_region(region: str, api_key: str) -> pd.DataFrame:
    """Fetch all hourly demand rows for one region, paging through results."""
    rows, offset = [], 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "D",  # D = demand
            "start": START,
            "end": END,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        resp = _get_with_retry(params)
        batch = resp.json()["response"]["data"]
        if not batch:
            break
        rows.extend(batch)
        print(f"  {region}: {len(rows):,} rows so far")
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)  # be polite to the API

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  WARNING: no data returned for {region}")
        return df
    df = df.rename(columns={"period": "timestamp_utc", "value": "load_mwh"})
    df["region"] = region
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["load_mwh"] = pd.to_numeric(df["load_mwh"], errors="coerce")
    return df[["timestamp_utc", "region", "load_mwh"]]


def _get_with_retry(params: dict, retries: int = 3) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            if attempt == retries:
                raise
            wait = 2 * attempt
            print(f"  request failed ({err}); retrying in {wait}s")
            time.sleep(wait)


def main() -> None:
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise SystemExit("Set the EIA_API_KEY environment variable first.")

    frames = []
    for region in REGIONS:
        print(f"Fetching {region}...")
        frames.append(fetch_region(region, api_key))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["region", "timestamp_utc"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print("\nQuick data-quality check (per region):")
    summary = df.groupby("region").agg(
        rows=("load_mwh", "size"),
        missing=("load_mwh", lambda s: s.isna().sum()),
        first=("timestamp_utc", "min"),
        last=("timestamp_utc", "max"),
        min_load=("load_mwh", "min"),
        max_load=("load_mwh", "max"),
    )
    print(summary)


if __name__ == "__main__":
    main()
