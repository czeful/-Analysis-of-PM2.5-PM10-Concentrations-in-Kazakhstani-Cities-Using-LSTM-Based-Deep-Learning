from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\airdata_article.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\airdata_article_2020_2024.csv")

START = pd.Timestamp("2020-01-01 00:00:00")
END = pd.Timestamp("2024-12-31 23:00:00")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

df = df[(df["datetime"] >= START) & (df["datetime"] <= END)].copy()
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

df.to_csv(OUT_FILE, index=False)

print("Done")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nRows by city:")
print(df["city"].value_counts())
print("\nDate range:")
print(df["datetime"].min(), "->", df["datetime"].max())
print("\nMissing values:")
print(df.isna().sum())
print("\nPreview:")
print(df.head())