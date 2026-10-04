# Grid Load Forecasting & Anomaly Agent

An AI-assisted system that forecasts hourly electricity demand for US grid regions, flags anomalies, and (in later milestones) uses a Claude agent to explain them in plain language.

**Status:** Milestone 1 complete (data pipeline, cleaning, baseline forecast). Models, anomaly detection, dashboard, and agent are in progress.

## Why this project

Grid operators need to anticipate demand and quickly understand when load deviates from what's expected. This project builds that workflow end to end using public data only: ingest, clean, forecast, detect anomalies, map results by region, and explain them through an agent.

## Regions

| Code | Region | Why it's included |
|------|--------|-------------------|
| `ERCO` | ERCOT (Texas) | Extreme heat swings and winter storm events |
| `CISO` | CAISO (California) | Heavy solar shapes the daily load curve |
| `DUK` | Duke Energy Carolinas | Mixed climate, a more "typical" region |
| `FPL` | Florida Power & Light | Hot and humid, with hurricane-related disruptions |

## Data

- **Load:** hourly demand from the [EIA Open Data API](https://www.eia.gov/opendata/) (v2), 2023-01-01 through 2025-12-31.
- All data is public. No proprietary or confidential data is used anywhere in this project.

### Data quality notes

Raw data is not used as-is. The cleaning step found and handled:

- **FPL spikes and collapses:** values like 66,000-72,000 MW (about 4x normal) and near-zero readings, treated as data errors rather than real events.
- **Missing data:** five whole-day gaps in FPL (four of 24 hours, one of 48 hours) and a handful of short gaps in CISO and DUK.
- **Cleaning rules:** values outside a wide band around each region's median (40%-200%) are flagged as errors. Only gaps of 6 hours or less are interpolated; longer gaps are left blank rather than invented. Every changed row is flagged, and the raw value is always kept.

## Baseline results

Seasonal naive baselines, scored on a 2025 holdout (8,760 hours per region, no gaps):

| Region | Baseline | MAPE (%) | RMSE (MW) |
|--------|----------|---------:|----------:|
| CISO | same hour yesterday | 4.88 | 1,746 |
| CISO | same hour last week | 5.86 | 2,223 |
| DUK | same hour yesterday | 7.05 | 1,267 |
| DUK | same hour last week | 12.73 | 2,221 |
| ERCO | same hour yesterday | 4.55 | 3,603 |
| ERCO | same hour last week | 8.70 | 6,682 |
| FPL | same hour yesterday | 4.73 | 1,107 |
| FPL | same hour last week | 9.96 | 2,181 |

"Same hour yesterday" wins in every region, which suggests day-to-day weather persistence matters more than weekly routine. That motivates adding weather features next. These scores are what later models need to beat.

## Project structure

```
.
├── pull_eia_load.py        # Step 1: download hourly load from EIA
├── clean_load.py           # Step 2: rebuild timeline, flag errors, repair short gaps
├── baseline_forecast.py    # Step 3: seasonal naive baselines + scoring
├── exploration/            # one-off inspection scripts
├── requirements.txt
└── data/                   # created when you run the scripts (not committed)
```

## Setup and usage

1. Get a free API key at [eia.gov/opendata](https://www.eia.gov/opendata/).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set the key as an environment variable (never put it in the code):

   PowerShell:
   ```
   $env:EIA_API_KEY="your_key_here"
   ```
   macOS / Linux:
   ```
   export EIA_API_KEY="your_key_here"
   ```
4. Run the pipeline in order, from the project folder:
   ```
   python pull_eia_load.py
   python clean_load.py
   python baseline_forecast.py
   ```

Outputs land in `data/raw/` and `data/processed/`.

## Roadmap

- [x] **Milestone 1:** data pipeline, cleaning, baseline forecast
- [ ] **Milestone 2:** weather features, improved model (Prophet or scikit-learn), anomaly detection on forecast residuals
- [ ] **Milestone 3:** FastAPI endpoints and Streamlit dashboard with regional map (GeoPandas + Folium)
- [ ] **Milestone 4:** Claude agent that calls the API to explain anomalies and answer questions about the grid
- [ ] **Phase 2 ideas:** ArcGIS Online export, more regions, scheduled refresh

## Tech stack

Python, pandas, requests (current). Planned: Prophet / scikit-learn, FastAPI, Streamlit, GeoPandas, Folium, Claude API.

## Author

Izayah Rahming