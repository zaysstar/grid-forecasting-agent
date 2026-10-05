# Grid Load Forecasting & Anomaly Agent

An AI-assisted system that forecasts hourly electricity demand for US grid regions, flags anomalies, and (in later milestones) uses a Claude agent to explain them in plain language.

**Status:** Milestones 1 and 2 complete (data pipeline, cleaning, baselines, weather-aware model, anomaly detection). Dashboard and agent are next.

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

### Data quality and cleaning

Raw data is not used as-is. The cleaning step found and handled:

- **FPL spikes and collapses:** readings around 66,000-72,000 MW (about 4x normal) and near-zero values, treated as data errors rather than real events.
- **Missing data:** five whole-day gaps in FPL (four of 24 hours, one of 48 hours) and a handful of short gaps in CISO and DUK.
- **In-range bad readings:** values that look plausible in isolation but are wrong in context, such as a one-hour drop to 11,819 MW in the middle of a 31,000 MW CISO summer afternoon, or a one-hour jump to 22,495 MW inside a steady FPL evening decline (about 16,900 MW expected).

Three rules flag data errors:

1. **Band check:** values outside 40%-200% of the region's median load.
2. **Neighbor check:** values more than 25% away from the median of the surrounding 7 hours.
3. **One-hour spike check:** an hour that is a local peak or trough and more than 20% away from the midpoint of its two neighbors. This catches spikes inside steep ramps, where the 7-hour median moves with the ramp and misses them.

Only gaps of 6 hours or less are interpolated. Longer gaps stay blank rather than invented. Every changed row is flagged, and the raw value is always kept.

**Lesson: forecast inputs inherit data errors.** The model uses past load as an input, so one bad reading corrupts two forecasts: the hour itself, and the same hour a day later (through the 24-hour input). The FPL reading above produced a real anomaly flag and a fake one the next day. Both were traced back to the data and removed by the third rule. The anomaly event list works as an audit of the data pipeline as well as of the grid.

## Forecasting methodology

- **Splits:** train 2023, validate 2024 (model selection), then train 2023-2024 and test 2025 (8,760 hours per region, no gaps).
- **Baselines:** seasonal naive, using load from the same hour yesterday (24h) and the same hour last week (168h).
- **Candidate models** (scikit-learn `HistGradientBoostingRegressor`, one per region):
  - **v1:** predicts load in MW directly.
  - **v2:** predicts load relative to the previous week's average level, then scales back to MW. Trees cannot predict outside the range seen in training, so v1 under-forecasts when demand grows.
  - **blend:** the average of v1 and v2.
- **Selection:** the best candidate per region is chosen on 2024 validation error, before the test year is scored. The choice was blend for CISO and ERCO, and v1 for DUK and FPL. In hindsight, each choice matched the best candidate on 2025.
- **Features:** hour, day of week, month, and US federal holiday flag (all in local time), temperature, humidity, 24-hour average temperature, and load from 24h and 168h earlier. Six major holidays (New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas) count as Sundays in the day-of-week feature, so the model reuses the weekend pattern instead of learning each holiday from two examples.
- **Scoring:** MAPE and RMSE on identical hours for the model and baselines.

## Forecasting results

MAPE (%) on the 2025 test year. Lower is better. The Model column is the validation-selected candidate.

| Region | Same hour last week | Same hour yesterday | Model | Chosen | Improvement vs. yesterday |
|--------|--------------------:|--------------------:|------:|:------:|--------------------------:|
| CISO | 5.84 | 4.86 | **2.90** | blend | 40.4% |
| DUK | 12.73 | 7.05 | **3.06** | v1 | 56.6% |
| ERCO | 8.70 | 4.55 | **2.88** | blend | 36.5% |
| FPL | 9.95 | 4.73 | **3.43** | v1 | 27.4% |

All three candidates on the test year (MAPE %):

| Region | v1 | v2 | blend |
|--------|---:|---:|------:|
| CISO | 2.97 | 3.04 | **2.90** |
| DUK | **3.06** | 4.95 | 3.62 |
| ERCO | 3.47 | 3.37 | **2.88** |
| FPL | **3.43** | 4.13 | 3.62 |

RMSE (MW) of the chosen model against the same-hour-yesterday baseline, and the chosen model's bias (mean of actual minus forecast; positive means the model runs low):

| Region | Baseline RMSE | Model RMSE | Model bias |
|--------|--------------:|-----------:|-----------:|
| CISO | 1,724 | 1,056 | 30 |
| DUK | 1,267 | 577 | 84 |
| ERCO | 3,603 | 2,238 | 645 |
| FPL | 1,103 | 825 | 123 |

**Takeaways**

- Same hour yesterday beats same hour last week in every region, so day-to-day weather persistence matters more than weekly routine.
- The chosen model beats the stronger baseline everywhere. The gain is largest in DUK, where the baseline was weakest.
- v2 alone is not an upgrade: it removed most of the upward drift in ERCO demand but hurt DUK and FPL. Averaging it with v1 gave the best ERCO and CISO results.
- ERCO started as the weakest region and ended among the strongest, though it still runs about 645 MW low on average. FPL now has the highest error (3.43% MAPE).

## Anomaly detection

Residual = actual load minus forecast, computed on the 2025 out-of-sample forecasts.

- **Scoring:** robust z-scores (median and median absolute deviation, so a few huge misses don't distort the cutoff), scaled within each calendar month so seasons where the model is normally less accurate are not over-flagged.
- **Two rules:** a single-hour spike rule, and a sustained rule on the trailing 6-hour average residual. An hour is flagged if either fires.
- **Operating threshold:** |z| > 4.0.
- **Events:** consecutive flagged hours (gaps up to 2 hours) are grouped into events, each with peak size, direction, and temperature context (temperature at the peak compared with the prior 30 days).

### Results at threshold 4.0

| Region | Typical error (MW) | Hours flagged | Events |
|--------|-------------------:|--------------:|-------:|
| CISO | 778 | 2.29% | 59 |
| DUK | 416 | 1.53% | 48 |
| ERCO | 1,697 | 0.81% | 16 |
| FPL | 589 | 1.77% | 27 |

The share of hours flagged on real data is an **upper bound** on the false-positive rate, since some flagged hours are genuine events and there are no labeled anomalies to check against.

### Injection test

Because real anomalies are unlabeled, detection ability is measured by adding synthetic spikes and dips (3-12 hours long, 5%, 10% or 20% of the forecast) to the 2025 data and checking whether they are flagged. Each region gets about 200 injected events per size, which gives roughly +/-7 percentage points of sampling error on each figure. Share caught at threshold 4.0 (both rules):

| Region | 5% size | 10% size | 20% size |
|--------|--------:|---------:|---------:|
| CISO | 10% | 62% | 100% |
| DUK | 9% | 39% | 95% |
| ERCO | 5% | 49% | 97% |
| FPL | 5% | 37% | 88% |

### Threshold trade-off (both rules)

| Region | Threshold | Hours flagged (%) | 5% size caught | 10% size caught | 20% size caught |
|--------|----------:|------------------:|---------------:|----------------:|----------------:|
| CISO | 3.0 | 6.80 | 32% | 88% | 100% |
| CISO | 3.5 | 4.09 | 20% | 75% | 100% |
| CISO | 4.0 | 2.29 | 10% | 62% | 100% |
| CISO | 5.0 | 0.68 | 3% | 33% | 98% |
| DUK | 3.0 | 4.65 | 31% | 72% | 98% |
| DUK | 3.5 | 2.89 | 18% | 53% | 97% |
| DUK | 4.0 | 1.53 | 9% | 39% | 95% |
| DUK | 5.0 | 0.45 | 4% | 27% | 83% |
| ERCO | 3.0 | 3.45 | 23% | 76% | 99% |
| ERCO | 3.5 | 1.78 | 12% | 63% | 98% |
| ERCO | 4.0 | 0.81 | 5% | 49% | 97% |
| ERCO | 5.0 | 0.22 | 1% | 22% | 90% |
| FPL | 3.0 | 5.27 | 25% | 74% | 97% |
| FPL | 3.5 | 2.95 | 14% | 55% | 92% |
| FPL | 4.0 | 1.77 | 5% | 37% | 88% |
| FPL | 5.0 | 0.89 | 2% | 23% | 79% |

**Takeaways**

- Detection sensitivity is limited by model error: a miss must be roughly 4 times the typical error to be flagged. A more accurate model makes the detector more sensitive.
- Large deviations (20% of load) are caught 88-100% of the time. Deviations around 10% are caught 37-62% of the time. Deviations around 5% are mostly missed (5-10%).
- Lowering the threshold to 3.0 raises recall at the 10% size to 72-88%, but flags 3.5-6.8% of all hours, which is too noisy for alerts. Threshold 4.0 is the chosen operating point.
- The sustained rule added at most 3 percentage points of recall, within sampling error, because forecast errors persist for hours and averaging does not reduce them. It is kept for completeness.
- Several of the largest events line up with temperature shifts: DUK on Dec 15 (17.0°C colder than the prior 30 days) and ERCO on Feb 19 (11.5°C colder). Most large events show no temperature signature.

### Investigated events

- **FPL single-hour spikes (May 9 and 10): resolved.** The May 9 reading was a data error that the first two cleaning rules missed (see Data quality). The May 10 flag was an echo: the actual load was smooth, and the forecast was contaminated by the bad May 9 input. Both disappeared after the third cleaning rule.
- **ERCO Memorial Day (May 26): partly explained.** The actual load was smooth, while the forecast followed a normal weekday shape. Treating major holidays as Sundays shrank the miss from 13 hours at -8,083 MW (mean residual) to 8 hours at -6,716 MW, but did not eliminate it.

### Open questions

- ERCO events on Jun 15 (-8,708 MW) and Jul 1 (-7,046 MW) have no weather or holiday explanation yet.
- DUK shows isolated single-hour spikes on Jun 30, Aug 1, and Sep 6 that have not been examined.

## Limitations

- The model uses **observed** weather as a stand-in for a weather forecast. A real day-ahead system would use forecasted weather, so results are somewhat optimistic.
- Weather comes from one city per region, a simplification of how weather varies across a grid.
- Flagged events are "the model was surprised" signals. They mix real grid events, remaining data errors, and model weaknesses, and there are no ground-truth labels.
- The injection test is synthetic. It measures sensitivity to known shifts, not performance on real operational anomalies.
- Model selection on a validation year can only anticipate drift that already appears in that year.

## Project structure

```
.
├── pull_eia_load.py        # Step 1: download hourly load from EIA
├── clean_load.py           # Step 2: rebuild timeline, flag errors, repair short gaps
├── baseline_forecast.py    # Step 3: seasonal naive baselines + scoring
├── pull_weather.py         # Step 4: download hourly weather from Open-Meteo
├── model_forecast.py       # Step 5: candidate models, validation selection, scoring
├── detect_anomalies.py     # Step 6: anomaly flags, events, injection test, threshold sweep
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
   python detect_anomalies.py
   ```

Outputs land in `data/raw/` and `data/processed/`. Key outputs: `model_forecasts.csv` (forecasts), `anomaly_flags.csv` (hourly scores and flags), `anomaly_events.csv` (grouped events with temperature context).

## Roadmap

- [x] **Milestone 1:** data pipeline, cleaning, baseline forecast
- [x] **Milestone 2:** weather data, validated model selection, anomaly detection with documented flag rate, recall, and threshold trade-off
- [ ] **Milestone 3:** FastAPI endpoints and Streamlit dashboard with regional map (GeoPandas + Folium)
- [ ] **Milestone 4:** Claude agent that calls the API to explain anomalies and answer questions about the grid
- [ ] **Phase 2 ideas:** ArcGIS Online export, forecasted (not observed) weather inputs, multiple weather stations per region, two-tier "watch" and "alert" thresholds, more regions, scheduled refresh

## Tech stack

Python, pandas, NumPy, scikit-learn, requests (current). Planned: FastAPI, Streamlit, GeoPandas, Folium, Claude API.

## Author

Izayah Rahming