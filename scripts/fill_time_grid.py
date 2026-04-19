from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\final_hourly_dataset.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\final_hourly_fullgrid.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

cities = df["city"].unique()

full_parts = []

for city in cities:
    print(f"Processing {city}")

    city_df = df[df["city"] == city].copy()

    start = city_df["datetime"].min().floor("h")
    end = city_df["datetime"].max().ceil("h")

    full_range = pd.date_range(start=start, end=end, freq="h")

    full_df = pd.DataFrame({
        "datetime": full_range,
        "city": city
    })

    merged = full_df.merge(
        city_df,
        on=["datetime", "city"],
        how="left"
    )

    full_parts.append(merged)

full_df = pd.concat(full_parts, ignore_index=True)
full_df = full_df.sort_values(["city", "datetime"]).reset_index(drop=True)

full_df.to_csv(OUT_FILE, index=False)

print("\nDONE")
print(f"Saved to: {OUT_FILE}")
print("Shape:", full_df.shape)

print("\nRows by city:")
print(full_df["city"].value_counts())

print("\nMissing values:")
print(full_df.isna().sum())