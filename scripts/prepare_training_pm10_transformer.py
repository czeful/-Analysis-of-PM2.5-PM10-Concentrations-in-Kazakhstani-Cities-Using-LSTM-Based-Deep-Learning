from pathlib import Path
import pandas as pd
import numpy as np

IN_FILE = Path(r"C:\air_quality_project\processed\final_features_pm25_upgrade.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\training_pm10_transformer.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)

required_cols = [
    "pm10",
    "pm10_lag_1", "pm10_lag_3", "pm10_lag_6", "pm10_lag_12", "pm10_lag_24",
    "pm10_roll_mean_3", "pm10_roll_mean_6", "pm10_roll_mean_12", "pm10_roll_mean_24",
    "pm10_roll_std_3", "pm10_roll_std_6", "pm10_roll_std_12", "pm10_roll_std_24",
    "delta_t", "blh_inv"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()

df = df[df["pm10"] >= 0].copy()
df = df[df["pm10"] <= 1500].copy()

df["pm10_log"] = np.log1p(df["pm10"])

df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

df.to_csv(OUT_FILE, index=False)

print("\nDONE")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df.shape)

print("\nRows by city:")
print(df["city"].value_counts())

print("\nDate range:")
print(df["datetime"].min(), "->", df["datetime"].max())

print("\nMissing values:")
print(df.isna().sum())

print("\nPreview:")
print(df.head())