from pathlib import Path
import pandas as pd

SENT_DIR = Path(r"C:\air_quality_project\raw\sentinel")
OUT_FILE = Path(r"C:\air_quality_project\processed\sentinel_combined.csv")

cities = ["almaty", "astana", "karaganda"]

city_frames = []

for city in cities:
    print(f"\nCity: {city}")

    files = {
        "aod_354": SENT_DIR / f"{city}_AOD_354.csv",
        "co_col": SENT_DIR / f"{city}_CO_col.csv",
        "no2_trop": SENT_DIR / f"{city}_NO2_trop.csv",
        "so2_col": SENT_DIR / f"{city}_SO2_col.csv",
    }

    frames = []

    for var, fp in files.items():
        if not fp.exists():
            print(f"SKIP missing file: {fp.name}")
            continue

        print(f"Reading {fp.name}")

        df = pd.read_csv(fp)

        if "date" not in df.columns:
            print(f"SKIP {fp.name} - no date column")
            continue

        # rename value column → standardized name
        value_col = [c for c in df.columns if c != "date"][0]

        temp = df.copy()
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.rename(columns={value_col: var})

        temp = temp[["date", var]]
        temp = temp.dropna(subset=["date"])

        frames.append(temp)

    if not frames:
        print(f"No data for {city}")
        continue

    city_df = frames[0]

    for other in frames[1:]:
        city_df = city_df.merge(other, on="date", how="outer", validate="one_to_one")

    city_df["city"] = city.capitalize()

    cols = ["date", "city"] + [c for c in city_df.columns if c not in ["date", "city"]]
    city_df = city_df[cols]

    city_df = city_df.sort_values("date").reset_index(drop=True)

    print(f"{city} shape: {city_df.shape}")
    print(f"{city} columns: {city_df.columns.tolist()}")

    city_frames.append(city_df)

if not city_frames:
    raise ValueError("No Sentinel data collected")

sentinel_df = pd.concat(city_frames, ignore_index=True)
sentinel_df = sentinel_df.sort_values(["city", "date"]).reset_index(drop=True)

sentinel_df.to_csv(OUT_FILE, index=False)

print("\nDone")
print(f"Saved to: {OUT_FILE}")
print(f"Shape: {sentinel_df.shape}")
print("\nColumns:")
print(sentinel_df.columns.tolist())
print("\nRows by city:")
print(sentinel_df['city'].value_counts())
print("\nMissing values:")
print(sentinel_df.isna().sum())
print("\nPreview:")
print(sentinel_df.head())