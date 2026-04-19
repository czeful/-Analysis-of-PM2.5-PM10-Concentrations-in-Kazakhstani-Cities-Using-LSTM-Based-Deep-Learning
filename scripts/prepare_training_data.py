from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\final_features.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\training_dataset.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)

# оставляем исходный dataset нетронутым
df_clean = df.copy()

# строки годятся для обучения только если:
# 1) есть target pm25
# 2) есть обязательные лаги для pm25
required_cols = [
    "pm25",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_24"
]

missing_required = [col for col in required_cols if col not in df_clean.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# удаляем только невалидные строки для обучения
df_clean = df_clean.dropna(subset=required_cols)

print("After drop rows with missing target/lags:", df_clean.shape)
print("Removed rows:", len(df) - len(df_clean))

# базовая sanity check для target
df_clean = df_clean[df_clean["pm25"] >= 0].copy()
df_clean = df_clean[df_clean["pm25"] <= 1000].copy()

# сортировка
df_clean = df_clean.sort_values(["city", "datetime"]).reset_index(drop=True)

# сохраняем в НОВЫЙ файл
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