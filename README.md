# Grid Load Forecasting & Anomaly Agent

An AI-assisted system that forecasts hourly electricity demand for US grid regions, flags anomalies, and (in later milestones) uses a Claude agent to explain them in plain language.

**Status:** Milestones 1 and most of 2 are complete (data pipeline, cleaning, baselines, weather-aware model). Anomaly detection, dashboard, and agent are in progress.

## Why this project

Grid operators need to anticipate demand and quickly understand when load deviates from what's expected. This project builds that workflow end to end using public data only: ingest, clean, forecast, detect anomalies, map results by region, and explain them through an agent.

## Regions

| Code | Region | Weather city | Why it's included |
|------|--------|--------------|-------------------|
| `ERCO` | ERCOT (Texas) | Dallas, TX | Extreme heat swings and winter storm events |
| `CISO` | CAISO (California) | Los Angeles, CA | Heavy solar shapes the daily load curve |
| `DUK` | Duke Energy Carolinas | Charlotte, NC | Mixed climate, a more "typical" region |
| `FPL` | Florida Power & Light | Miami, FL | Hot and humid, with hurricane-related disruptions |

## Data

- **Load:** hourly demand from the [EIA Open Data API](https://www.eia.gov/opendata/) (v2), 2023-01-01 through 2025-12-31.
- **Weather:** hourly temperature and humidity from the [Open-Meteo](https://open-meteo.com/) historical archive, one representative city per region.
- All data is public. No proprietary or confidential data is used anywhere in this project.

### Data quality notes

Raw data is not used as-is. The cleaning step found and handled:

- **FPL spikes and collapses:** values like 66,000-72,000 MW (about 4x normal) and near-zero readings, treated as data errors rather than real events.
- **Missing data:** five whole-day gaps in FPL (four of 24 hours, one of 48 hours) and a handful of short gaps in CISO and DUK.
- **Cleaning rules:** values outside a wide band around each region's median (40%-200%) are flagged as errors. Only gaps of 6 hours or less are interpolated; longer gaps are left blank rather than invented. Every changed row is flagged, and the raw value is always kept.

## Methodology

- **Holdout:** train on 2023-2024, test on 2025 (8,760 hours per region, no gaps in the test year).
- **Baselines:** seasonal naive, using the load from the same hour yesterday (24h) and the same hour last week (168h).
- **Model:** gradient boosting (scikit-learn `HistGradientBoostingRegressor`), one model per region.
- **Features:** hour, day of week, month, and US holiday flag (all in local time), temperature, humidity, 24-hour average temperature, and load from 24h and 168h earlier.
- **Scoring:** MAPE and RMSE, computed on the same hours for the model and baselines so the comparison is fair.

## Results

MAPE (%) on the 2025 holdout. Lower is better.

| Region | Same hour last week | Same hour yesterday | Model | Improvement vs. yesterday |
|--------|--------------------:|--------------------:|------:|--------------------------:|
| CISO | 5.86 | 4.88 | **2.98** | 39.0% |
| DUK | 12.73 | 7.05 | **3.10** | 56.1% |
| ERCO | 8.70 | 4.55 | **3.51** | 22.7% |
| FPL | 9.96 | 4.73 | **3.47** | 26.8% |

RMSE (MW), model vs. the same-hour-yesterday baseline:

| Region | Baseline | Model |
|--------|---------:|------:|
| CISO | 1,746 | 1,099 |
| DUK | 1,267 | 583 |
| ERCO | 3,603 | 2,641 |
| FPL | 1,107 | 828 |

**Takeaways**

- Same hour yesterday beats same hour last week in every region, which suggests day-to-day weather persistence matters more than weekly routine.
- The model beats the stronger baseline everywhere. The gain is largest in DUK, where the baseline was weakest.
- ERCO shows the smallest improvement and the largest absolute error. Its extreme weather swings are one possible cause, but this has not been investigated yet.

### Limitation

The model uses **observed** weather as a stand-in for a weather forecast. A real day-ahead system would use forecasted weather, so these results are somewhat optimistic compared with a real deployment.

## Project structure

```
.
├── pull_eia_load.py        # Step 1: download hourly load from EIA
├── clean_load.py           # Step 2: rebuild timeline, flag errors, repair short gaps
├── baseline_forecast.py    # Step 3: seasonal naive baselines + scoring
├── pull_weather.py         # Step 4: download hourly weather from Open-Meteo
├── model_forecast.py       # Step 5: weather-aware model, scored against baselines
├── exploration/            # one-off inspection scripts
├── requirements.txt
└── data/                   # created when you run the scripts (not committed)
```

## Setup and usage

1. Get a free EIA API key at [eia.gov/opendata](https://www.eia.gov/opendata/). Open-Meteo needs no key.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set the EIA key as an environment variable (never put it in the code):

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
   python pull_weather.py
   python model_forecast.py
   ```

Outputs land in `data/raw/` and `data/processed/`.

## Roadmap

- [x] **Milestone 1:** data pipeline, cleaning, baseline forecast
- [x] **Milestone 2a:** weather data and improved model (beats baselines in all four regions)
- [ ] **Milestone 2b:** anomaly detection on forecast residuals, with false-positive rate documented
- [ ] **Milestone 3:** FastAPI endpoints and Streamlit dashboard with regional map (GeoPandas + Folium)
- [ ] **Milestone 4:** Claude agent that calls the API to explain anomalies and answer questions about the grid
- [ ] **Phase 2 ideas:** ArcGIS Online export, forecasted (not observed) weather inputs, multiple weather stations per region, more regions, scheduled refresh

## Tech stack

Python, pandas, NumPy, scikit-learn, requests (current). Planned: FastAPI, Streamlit, GeoPandas, Folium, Claude API.

## Author

Izayah Rahming