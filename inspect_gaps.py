import pandas as pd

df = pd.read_csv("data/processed/load_clean.csv", parse_dates=["timestamp_utc"])
fpl = df[df.region == "FPL"].set_index("timestamp_utc")

# the 14 flagged outliers
print(fpl.loc[fpl.flag_outlier, ["load_mwh_raw"]])

# remaining gaps as start / end / length
gap = fpl.flag_still_missing
run = (gap != gap.shift()).cumsum()
gaps = (
    fpl[gap].reset_index()
    .groupby(run[gap].values)
    .agg(start=("timestamp_utc", "min"), end=("timestamp_utc", "max"), hours=("timestamp_utc", "size"))
)
print(gaps)