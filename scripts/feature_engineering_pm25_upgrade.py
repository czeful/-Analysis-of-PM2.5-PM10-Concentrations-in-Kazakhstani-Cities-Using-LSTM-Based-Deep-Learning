from pathlib import Path
import pandas as pd
import numpy as np

IN_FILE = Path(r"C:\air_quality_project\processed\final_hourly_clean.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\final_features_pm25_upgrade.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

parts = []

for city in df["city"].unique():
    print(f"Processing {city}")
    city_df = df[df["city"] == city].copy().sort_values("datetime").reset_index(drop=True)

    # lags
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        city_df[f"pm25_lag_{lag}"] = city_df["pm25"].shift(lag)
        city_df[f"pm10_lag_{lag}"] = city_df["pm10"].shift(lag)

    # rolling stats for PM2.5
    for window in [3, 6, 12, 24]:
        city_df[f"pm25_roll_mean_{window}"] = city_df["pm25"].rolling(window).mean()
        city_df[f"pm25_roll_std_{window}"] = city_df["pm25"].rolling(window).std()

    # rolling stats for PM10
    for window in [3, 6, 12, 24]:
        city_df[f"pm10_roll_mean_{window}"] = city_df["pm10"].rolling(window).mean()
        city_df[f"pm10_roll_std_{window}"] = city_df["pm10"].rolling(window).std()

    # wind features
    city_df["wind_speed"] = np.sqrt(city_df["u10"] ** 2 + city_df["v10"] ** 2)
    city_df["wind_dir"] = np.degrees(np.arctan2(city_df["v10"], city_df["u10"]))
    city_df["wind_dir"] = (city_df["wind_dir"] + 360) % 360

    # physical interaction features
    city_df["delta_t"] = city_df["t850"] - city_df["t2m"]
    city_df["blh_inv"] = 1.0 / (city_df["blh"] + 1.0)
    city_df["so2col_wind"] = city_df["so2_col"] * city_df["wind_speed"]
    city_df["co_blhinv"] = city_df["co"] * city_df["blh_inv"]

    # time features
    city_df["hour"] = city_df["datetime"].dt.hour
    city_df["month"] = city_df["datetime"].dt.month
    city_df["dayofyear"] = city_df["datetime"].dt.dayofyear

    city_df["hour_sin"] = np.sin(2 * np.pi * city_df["hour"] / 24)
    city_df["hour_cos"] = np.cos(2 * np.pi * city_df["hour"] / 24)

    city_df["month_sin"] = np.sin(2 * np.pi * city_df["month"] / 12)
    city_df["month_cos"] = np.cos(2 * np.pi * city_df["month"] / 12)

    city_df["dayofyear_sin"] = np.sin(2 * np.pi * city_df["dayofyear"] / 365.25)
    city_df["dayofyear_cos"] = np.cos(2 * np.pi * city_df["dayofyear"] / 365.25)

    parts.append(city_df)

df_out = pd.concat(parts, ignore_index=True)
df_out = df_out.sort_values(["city", "datetime"]).reset_index(drop=True)

df_out.to_csv(OUT_FILE, index=False)

print("\nDONE")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df_out.shape)
print("\nColumns:")
print(df_out.columns.tolist())
print("\nMissing values:")
print(df_out.isna().sum())
print("\nPreview:")
print(df_out.head())