from pathlib import Path
import pandas as pd
import numpy as np

IN_FILE = Path(r"C:\air_quality_project\processed\final_hourly_clean.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\final_features.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

result_parts = []

for city in df["city"].unique():
    print(f"Processing {city}")

    city_df = df[df["city"] == city].copy()

    # ======================
    # 1. LAG FEATURES
    # ======================
    for lag in [1, 3, 24]:
        city_df[f"pm25_lag_{lag}"] = city_df["pm25"].shift(lag)
        city_df[f"pm10_lag_{lag}"] = city_df["pm10"].shift(lag)

    # ======================
    # 2. WIND FEATURES
    # ======================
    city_df["wind_speed"] = np.sqrt(city_df["u10"]**2 + city_df["v10"]**2)

    # optional wind direction
    city_df["wind_dir"] = np.degrees(np.arctan2(city_df["v10"], city_df["u10"]))
    city_df["wind_dir"] = (city_df["wind_dir"] + 360) % 360

    # ======================
    # 3. TIME FEATURES
    # ======================
    city_df["hour"] = city_df["datetime"].dt.hour
    city_df["month"] = city_df["datetime"].dt.month

    # cyclic encoding
    city_df["hour_sin"] = np.sin(2 * np.pi * city_df["hour"] / 24)
    city_df["hour_cos"] = np.cos(2 * np.pi * city_df["hour"] / 24)

    city_df["month_sin"] = np.sin(2 * np.pi * city_df["month"] / 12)
    city_df["month_cos"] = np.cos(2 * np.pi * city_df["month"] / 12)

    # season
    def get_season(m):
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "spring"
        elif m in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    city_df["season"] = city_df["month"].apply(get_season)

    result_parts.append(city_df)

df_feat = pd.concat(result_parts, ignore_index=True)
df_feat = df_feat.sort_values(["city", "datetime"]).reset_index(drop=True)

df_feat.to_csv(OUT_FILE, index=False)

print("\nDONE")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df_feat.shape)

print("\nColumns:")
print(df_feat.columns.tolist())

print("\nMissing values:")
print(df_feat.isna().sum())

print("\nPreview:")
print(df_feat.head())