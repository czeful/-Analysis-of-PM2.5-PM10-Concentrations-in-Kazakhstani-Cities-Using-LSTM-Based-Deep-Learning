from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\final_features.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\training_dataset_joint_pm25_pm10.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)

df_clean = df.copy()

required_cols = [
    "pm25", "pm10",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_24",
    "pm10_lag_1", "pm10_lag_3", "pm10_lag_24"
]

missing_required = [col for col in required_cols if col not in df_clean.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

df_clean = df_clean.dropna(subset=required_cols)

print("After drop rows with missing joint targets/lags:", df_clean.shape)
print("Removed rows:", len(df) - len(df_clean))

# базовая sanity check
for col in ["pm25", "pm10"]:
    df_clean = df_clean[df_clean[col] >= 0].copy()
    df_clean = df_clean[df_clean[col] <= 1000].copy()

df_clean = df_clean.sort_values(["city", "datetime"]).reset_index(drop=True)

df_clean.to_csv(OUT_FILE, index=False)

print("\nDONE")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df_clean.shape)

print("\nRows by city:")
print(df_clean["city"].value_counts())

print("\nMissing values:")
print(df_clean.isna().sum())

print("\nDate range:")
print(df_clean["datetime"].min(), "->", df_clean["datetime"].max())

print("\nPreview:")
print(df_clean.head())