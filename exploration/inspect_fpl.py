import pandas as pd

df = pd.read_csv("data/raw/eia_hourly_load.csv", parse_dates=["timestamp_utc"])
fpl = df[df.region == "FPL"].set_index("timestamp_utc")["load_mwh"]

print(fpl.describe())
print(fpl[(fpl < 5000) | (fpl > 35000)])  # suspicious values

full = pd.date_range(fpl.index.min(), fpl.index.max(), freq="h")
print(full.difference(fpl.index))  # missing timestamps