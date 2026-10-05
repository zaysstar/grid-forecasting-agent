import pandas as pd

flags = pd.read_csv("data/processed/anomaly_flags.csv", parse_dates=["timestamp_utc"])
clean = pd.read_csv("data/processed/load_clean.csv", parse_dates=["timestamp_utc"])
df = flags.merge(clean[["region", "timestamp_utc", "load_mwh_raw"]], on=["region", "timestamp_utc"], how="left")

CHECKS = [("FPL", "2025-05-09 04:00"), ("FPL", "2025-05-10 04:00"), ("ERCO", "2025-05-26 18:00")]
for region, ts in CHECKS:
    t = pd.Timestamp(ts, tz="UTC")
    w = df[(df.region == region) & (df.timestamp_utc.between(t - pd.Timedelta(hours=5), t + pd.Timedelta(hours=5)))]
    print(f"\n{region} around {ts} UTC")
    print(w[["timestamp_utc", "load_mwh_raw", "actual", "pred_model", "residual", "temp_c"]].round(0).to_string(index=False))