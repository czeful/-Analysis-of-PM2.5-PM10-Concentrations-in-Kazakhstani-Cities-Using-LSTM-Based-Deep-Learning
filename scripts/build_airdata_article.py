import pandas as pd
from pathlib import Path

IN_FILE = Path(r"C:\air_quality_project\processed\airdata_combined.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\airdata_article.csv")

keep_cols = ["datetime", "city", "pm25", "pm10", "co", "no2", "so2"]

df = pd.read_csv(IN_FILE)

existing_keep_cols = [c for c in keep_cols if c in df.columns]
missing_cols = [c for c in keep_cols if c not in df.columns]

print("Keeping columns:", existing_keep_cols)
print("Missing columns:", missing_cols)

df = df[existing_keep_cols].copy()
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

df.to_csv(OUT_FILE, index=False)

print("\nDone")
print(f"Saved to: {OUT_FILE}")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())
print("\nRows by city:")
print(df['city'].value_counts())
print("\nMissing values:")
print(df.isna().sum())
print("\nPreview:")
print(df.head())