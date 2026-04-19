Вот полностью отформатированный текст для твоего `README.md`, объединяющий все разделы в едином, чистом стиле Markdown:

# Feature Attribution Analysis of PM2.5 and PM10 Concentrations in Kazakhstani Cities Using City-Specific LSTM–Transformer Models

This repository contains the full codebase, processed data workflow, trained model outputs, and feature attribution pipeline for a comparative urban air quality modelling study in three Kazakhstani cities: **Almaty**, **Astana**, and **Karaganda**.

The project integrates ground-based air pollution observations, ERA5 meteorological reanalysis, Sentinel-5P satellite products, and static urban morphology indicators to model PM2.5 and PM10 concentrations using **city-specific LSTM–Transformer** architectures.

---

## Study Overview

The project focuses on three contrasting urban regimes in Kazakhstan:
* **Almaty** — Enclosed basin city with strong accumulation and inversion conditions.
* **Astana** — Open-steppe city with strong transport sensitivity and episodic extreme events.
* **Karaganda** — Industrial city with mixed combustion and wind-modulated pollution structure.

### Modeling Framework
* **City-Specific Approach:** Separate models for each city and each pollutant (**PM2.5** and **PM10**).
* **Architecture:** Hybrid LSTM–Transformer.
* **Parameters:** 72-hour look-back window, Log-transformed targets, cyclic temporal encoding.
* **Interpretation:** SHAP (GradientExplainer) for post-hoc global feature attribution.
* **Robustness Note:** A robustness-filtered specification is used for **Astana PM2.5**, where rare extreme events (>200 µg/m³) are excluded from the final evaluation.

---

## Repository Structure

```text
.
├── raw/                         # Original datasets
│   ├── air_data/                # Ground stations
│   ├── era5/                    # Meteo reanalysis
│   ├── sentinel/                # Satellite products
│   └── morphology/              # Static urban indicators
├── processed/                   # Intermediate & final data
│   ├── era5_unpacked/
│   ├── sequence_data_.../       # Tensors for training
│   ├── models_.../              # Saved model weights
│   ├── shap_.../                # Computed SHAP values
│   └── final_outputs/           # Resulting tables/plots
├── scripts/                     # Implementation logic
│   ├── data_prep/               # Merging & cleaning
│   ├── engineering/             # Feature creation
│   ├── training/                # Model loops
│   └── analysis/                # SHAP & plotting
├── article.pdf                  # Associated manuscript
└── README.md
```

---

## Data Sources

* **Ground-level Air Quality:** Hourly city-level observations (PM2.5, PM10, CO, NO2, SO2) from *AirData.kz*.
* **ERA5 Reanalysis:** Meteorological predictors (`t2m`, `u10`, `v10`, `blh`, `tcwv`, `ssrd`, `msl`, `t850`).
* **Sentinel-5P:** Daily satellite predictors expanded to hourly (`AOD_354`, `NO2_trop`, `SO2_col`, `CO_col`).
* **Static Morphology:** City-level indicators (`TRI`, `NDVI`, `RND`, `TCI`, `D_ind`).

---

## Feature Engineering & Design

We utilize several feature groups to capture complex urban dynamics:
* **Lags:** 1, 3, 6, 12, 24, 48, 72 hours.
* **Statistics:** Rolling mean and standard deviation.
* **Temporal:** Cyclic encoding (`sin`/`cos`) for hour, month, and day of year.
* **Physical Interactions:** `wind_speed`, `wind_dir`, `delta_t`, `blh_inv`, `so2col_wind`, `co_blhinv`.

### Data Split
* **Train:** Start of record — 2023-06-30
* **Validation:** 2023-07-01 — 2023-12-31
* **Test:** 2024-01-01 — 2024-12-31
> **Note:** Scaling parameters are computed on the training set only to prevent data leakage.

---

## Main Scripts & Usage

### Core Pipeline
1. **Feature Engineering:** `python scripts/feature_engineering_pm25_upgrade.py`
2. **Training (Example for Almaty):**
   ```bash
   python scripts/prepare_sequences_pm25_by_city.py
   python scripts/train_pm25_transformer_city.py Almaty
   ```
3. **SHAP Analysis:**
   ```bash
   python scripts/run_shap_final_city.py Almaty pm25
   ```

---

## Final Test Performance

| City | Pollutant | R² | RMSE (µg/m³) | MAE (µg/m³) |
| :--- | :--- | :--- | :--- | :--- |
| **Almaty** | PM2.5 | 0.7218 | 7.04 | 4.35 |
| **Astana** | PM2.5* | 0.8811 | 12.69 | 7.23 |
| **Karaganda** | PM2.5 | 0.6201 | 11.30 | 7.62 |
| **Almaty** | PM10 | 0.8234 | 11.24 | 7.47 |
| **Astana** | PM10 | 0.4594 | 84.27 | 22.64 |
| **Karaganda** | PM10 | 0.6371 | 10.40 | 7.09 |

*\*Astana PM2.5 results use robustness-filtered data (excluding >200 µg/m³).*

---

## Environment Setup

**Requirements:** Python 3.11+, PyTorch, SHAP, xarray.

```bash
pip install torch numpy pandas scikit-learn matplotlib shap joblib xarray netCDF4 cfgrib
```

---

## Limitations & Reproducibility

* **Scale:** Modelling is performed at city-aggregate level rather than individual stations.
* **Time-Series Integrity:** All splits are chronological to respect temporal dependencies.
* **Interpretation:** SHAP analysis represents global attribution across the test period.

---

## Citation

```text
[Authors]. Feature Attribution Analysis of PM2.5 and PM10 Concentrations in Kazakhstani Cities 
Using City-Specific LSTM–Transformer Models. [Journal, Year].
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
