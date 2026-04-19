from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\sentinel_combined.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\sentinel_hourly.csv")

df = pd.read_csv(IN_FILE)
df["date"] = pd.to_datetime(df["date"])

hourly_parts = []

for _, row in df.iterrows():
    base_date = row["date"]
    city = row["city"]

    for hour in range(24):
        new_row = row.copy()
        new_row["datetime"] = base_date + pd.Timedelta(hours=hour)
        hourly_parts.append(new_row)

hourly_df = pd.DataFrame(hourly_parts)

# remove original daily date column and keep datetime first
hourly_df = hourly_df.drop(columns=["date"])
cols = ["datetime", "city"] + [c for c in hourly_df.columns if c not in ["datetime", "city"]]
hourly_df = hourly_df[cols]

hourly_df = hourly_df.sort_values(["city", "datetime"]).reset_index(drop=True)

hourly_df.to_csv(OUT_FILE, index=False)

print("Done")
print(f"Saved to: {OUT_FILE}")
print("Shape:", hourly_df.shape)
print("\nColumns:")
print(hourly_df.columns.tolist())
print("\nRows by city:")
print(hourly_df["city"].value_counts())
print("\nDate range:")
print(hourly_df["datetime"].min(), "->", hourly_df["datetime"].max())
print("\nMissing values:")
print(hourly_df.isna().sum())
print("\nPreview:")
print(hourly_df.head())