# Feature Attribution Analysis of PM2.5 and PM10 Concentrations in Kazakhstani Cities Using City-Specific LSTM–Transformer Models

This repository contains the full codebase, processed data workflow, trained model outputs, and feature attribution pipeline for a comparative urban air quality modelling study in three Kazakhstani cities: **Almaty**, **Astana**, and **Karaganda**.  

The project integrates ground-based air pollution observations, ERA5 meteorological reanalysis, Sentinel-5P satellite products, and static urban morphology indicators to model PM2.5 and PM10 concentrations using **city-specific LSTM–Transformer** architectures.

The repository is intended to support:
- Computational reproducibility of the study
- Transparent access to preprocessing and modelling code
- Inspection of trained model outputs and SHAP-based feature attribution
- Reuse of the workflow in related urban air quality studies

## Study Overview

The project focuses on three contrasting urban regimes in Kazakhstan:

- **Almaty** — enclosed basin city with strong accumulation and inversion conditions
- **Astana** — open-steppe city with strong transport sensitivity and episodic extreme events
- **Karaganda** — industrial city with mixed combustion and wind-modulated pollution structure

The final modelling framework uses:
- Separate models for each city
- Separate models for each pollutant (**PM2.5** and **PM10**)
- 72-hour look-back window
- Log-transformed targets
- Rolling pollutant statistics
- Cyclic temporal encoding
- Physically informed interaction features
- SHAP for post-hoc feature attribution

A **robustness-filtered** specification is used for **Astana PM2.5**, where rare extreme events above 200 µg/m³ are excluded from the final model evaluation. This is reported transparently as a robustness analysis rather than a universal preprocessing rule.

## Repository Structure

```text
.
├── raw/
│   ├── air_data/
│   ├── era5/
│   ├── sentinel/
│   └── morphology/
│
├── processed/
│   ├── era5_unpacked/
│   ├── sequence_data_by_city/
│   ├── sequence_data_pm25_by_city/
│   ├── sequence_data_pm10_by_city/
│   ├── models_by_city/
│   ├── models_pm25_by_city/
│   ├── models_pm10_by_city/
│   ├── shap_by_city/
│   ├── shap_final/
│   └── final_outputs/
│
├── scripts/
│   ├── data preparation and merging scripts
│   ├── feature engineering scripts
│   ├── sequence preparation scripts
│   ├── model training scripts
│   ├── SHAP analysis scripts
│   └── plotting and reporting scripts
│
├── article.pdf
└── README.md
Data Sources
The project combines four main data groups:

Ground-level air quality data
Hourly city-level PM and gaseous pollutant observations from AirData.kz.
Main targets: PM2.5, PM10
Additional variables: CO, NO2, SO2
ERA5 meteorological reanalysis
Hourly predictors: t2m, u10, v10, blh, tcwv, ssrd, msl, t850
Sentinel-5P satellite data
Daily city-level satellite predictors expanded to hourly timestamps:
AOD_354, NO2_trop, SO2_col, CO_col
Static morphology features
City-level static predictors: TRI, NDVI, RND, TCI, D_ind

Final Modelling Design
The final study results are based on city-specific and pollutant-specific models:

Almaty PM2.5
Astana PM2.5 (robustness-filtered)
Karaganda PM2.5
Almaty PM10
Astana PM10
Karaganda PM10

Key settings:

72-hour look-back window
log1p target transformation
Huber Loss
Standardization fitted on training data only
Chronological train/validation/test split
SHAP GradientExplainer for global feature attribution

Final Feature Groups

ERA5 meteorological predictors
Satellite-derived aerosol and trace-gas indicators
Static morphology features
Lag features (1, 3, 6, 12, 24, 48, 72 hours)
Rolling mean and rolling standard deviation
Cyclic time features (hour_sin/cos, month_sin/cos, dayofyear_sin/cos)
Physical interaction terms (wind_speed, wind_dir, delta_t, blh_inv, so2col_wind, co_blhinv)

Final Data Split

Train: start of record — 2023-06-30
Validation: 2023-07-01 — 2023-12-31
Test: 2024-01-01 — 2024-12-31

All scaling parameters are computed using the training set only.
Main Scripts
A. Feature Engineering

feature_engineering_pm25_upgrade.py

B. PM2.5 Pipeline

prepare_training_pm25_transformer.py
split_pm25_transformer.py
prepare_sequences_pm25_by_city.py
train_pm25_transformer_city.py

C. PM10 Pipeline

prepare_training_pm10_transformer.py
split_pm10_transformer.py
prepare_sequences_pm10_by_city.py
train_pm10_transformer_city.py

D. Astana PM2.5 Robustness Workflow

analyze_astana_pm25_errors.py
filter_astana_extremes.py

E. SHAP Analysis

run_shap_final_city.py

Usage example:
Bashpython scripts/run_shap_final_city.py Almaty pm25
python scripts/run_shap_final_city.py Astana pm25
python scripts/run_shap_final_city.py Karaganda pm25
python scripts/run_shap_final_city.py Almaty pm10
# ... и т.д.
F. Final Reporting

make_final_metrics_table.py
plot_final_model_metrics.py
plot_final_shap_comparison.py

Example Workflow
Bash# 1. Feature engineering
python scripts/feature_engineering_pm25_upgrade.py

# 2. PM2.5 training
python scripts/prepare_training_pm25_transformer.py
python scripts/split_pm25_transformer.py
python scripts/prepare_sequences_pm25_by_city.py
python scripts/train_pm25_transformer_city.py Almaty
python scripts/train_pm25_transformer_city.py Astana
python scripts/train_pm25_transformer_city.py Karaganda

# 3. PM10 training (аналогично)
# 4. SHAP analysis
# 5. Generate tables and plots
Final Test Performance


CityPollutantR²RMSE (µg/m³)MAE (µg/m³)AlmatyPM2.50.72187.044.35AstanaPM2.50.881112.697.23KaragandaPM2.50.620111.307.62AlmatyPM100.823411.247.47AstanaPM100.459484.2722.64KaragandaPM100.637110.407.09
Важное примечание: Результаты для Astana PM2.5 получены на robustness-filtered выборке (исключены редкие экстремальные события > 200 µg/m³). Это целевой анализ устойчивости, а не универсальное правило предобработки.
SHAP Summary
Финальный SHAP-анализ выявил разные иерархии драйверов в зависимости от города:

Almaty: короткосрочная память накопления, переменные горения, суточный цикл, инверсионные взаимодействия граничного слоя.
Astana: серосодержащие предикторы, ветер, давление, термическая структура, суточная динамика.
Karaganda: индикаторы горения, persistence загрязнения, модуляция ветром.

Эти паттерны подтверждают главный научный вывод исследования: предсказуемость загрязнителей и иерархия факторов сильно зависят от урбанистического режима (бассейновый, степной, промышленный).
Environment
Recommended:

Python 3.11+
PyTorch
NumPy, pandas, scikit-learn
xarray, netCDF4, cfgrib
shap, matplotlib, joblib

Bashpip install numpy pandas scikit-learn matplotlib shap joblib xarray netCDF4 cfgrib torch
Reproducibility Notes
