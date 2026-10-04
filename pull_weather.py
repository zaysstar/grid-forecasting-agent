"""
Milestone 2, Step 1: Pull hourly weather for one representative city per region.

Source: Open-Meteo historical archive API (free, no API key needed).

Run from the project folder:  python pull_weather.py

Output:
    data/raw/weather_hourly.csv
    columns: timestamp_utc, region, temp_c, humidity_pct

Note: one city per region is a simplification. Load across a whole grid region
responds to weather in many places. Averaging several cities is a possible
later upgrade.
"""

import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2023-01-01"
END = "2025-12-31"
OUT_PATH = Path("data/raw/weather_hourly.csv")

# region code -> (representative city, latitude, longitude)
LOCATIONS = {
    "ERCO": ("Dallas, TX", 32.78, -96.80),
    "CISO": ("Los Angeles, CA", 34.05, -118.24),
    "DUK": ("Charlotte, NC", 35.23, -80.84),
    "FPL": ("Miami, FL", 25.76, -80.19),
}


def fetch_weather(region: str, city: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START,
        "end_date": END,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "GMT",  # match the UTC timestamps in the load data
    }
    for attempt in range(1, 4):
        try:
            resp = requests.get(API_URL, params=params, timeout=90)
            resp.raise_for_status()
            break
        except requests.RequestException as err:
            if attempt == 3:
                raise
            print(f"  request failed ({err}); retrying in {2 * attempt}s")
            time.sleep(2 * attempt)

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(hourly["time"], utc=True),
            "region": region,
            "temp_c": hourly["temperature_2m"],
            "humidity_pct": hourly["relative_humidity_2m"],
        }
    )
    print(f"  {region} ({city}): {len(df):,} rows")
    return df


def main() -> None:
    frames = []
    for region, (city, lat, lon) in LOCATIONS.items():
        print(f"Fetching {region}...")
        frames.append(fetch_weather(region, city, lat, lon))
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}\n")

    summary = df.groupby("region").agg(
        rows=("temp_c", "size"),
        missing_temp=("temp_c", lambda s: s.isna().sum()),
        missing_humidity=("humidity_pct", lambda s: s.isna().sum()),
        first=("timestamp_utc", "min"),
        last=("timestamp_utc", "max"),
        min_temp_c=("temp_c", "min"),
        max_temp_c=("temp_c", "max"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()