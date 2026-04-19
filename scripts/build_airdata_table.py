from pathlib import Path
import pandas as pd

AIR_DIR = Path(r"C:\air_quality_project\raw\air_data")
OUT_FILE = Path(r"C:\air_quality_project\processed\airdata_combined.csv")

CITY_MAP = {
    "al": "Almaty",
    "as": "Astana",
    "krg": "Karaganda",
}

city_frames = []

for city_code, city_name in CITY_MAP.items():
    city_dir = AIR_DIR / city_code

    if not city_dir.exists():
        print(f"SKIP city folder not found: {city_dir}")
        continue

    pollutant_frames = []

    csv_files = sorted(city_dir.glob("*.csv"))
    print(f"\nCity: {city_name}")
    print(f"Found files: {[f.name for f in csv_files]}")

    for fp in csv_files:
        pollutant = fp.stem.lower()
        print(f"Reading {city_name} - {pollutant}: {fp.name}")

        df = pd.read_csv(fp)

        required_cols = ["datetime_utc", "value_ugm3"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"SKIP {fp.name} - missing columns: {missing}")
            continue

        temp = df[["datetime_utc", "value_ugm3"]].copy()

        temp["datetime"] = pd.to_datetime(temp["datetime_utc"], errors="coerce", utc=True)

        # convert timezone-aware datetimes to naive UTC datetimes
        temp["datetime"] = temp["datetime"].dt.tz_localize(None)

        temp["value_ugm3"] = pd.to_numeric(temp["value_ugm3"], errors="coerce")

        temp = temp.dropna(subset=["datetime", "value_ugm3"])

        # city-level hourly mean across all stations
        temp = (
            temp.groupby("datetime", as_index=False)["value_ugm3"]
            .mean()
            .rename(columns={"value_ugm3": pollutant})
        )

        pollutant_frames.append(temp)

    if not pollutant_frames:
        print(f"No valid pollutant files for {city_name}")
        continue

    city_df = pollutant_frames[0]

    for other in pollutant_frames[1:]:
        city_df = city_df.merge(other, on="datetime", how="outer", validate="one_to_one")

    city_df["city"] = city_name

    cols = ["datetime", "city"] + [c for c in city_df.columns if c not in ["datetime", "city"]]
    city_df = city_df[cols]

    city_df = city_df.sort_values("datetime").reset_index(drop=True)

    print(f"{city_name} shape: {city_df.shape}")
    print(f"{city_name} columns: {city_df.columns.tolist()}")

    city_frames.append(city_df)

if not city_frames:
    raise ValueError("No AirData tables were created")

airdata_df = pd.concat(city_frames, ignore_index=True)
airdata_df = airdata_df.sort_values(["city", "datetime"]).reset_index(drop=True)

airdata_df.to_csv(OUT_FILE, index=False)

print("\nDone")
print(f"Saved to: {OUT_FILE}")
print(f"Shape: {airdata_df.shape}")
print("\nColumns:")
print(airdata_df.columns.tolist())
print("\nRows by city:")
print(airdata_df['city'].value_counts())
print("\nMissing values:")
print(airdata_df.isna().sum())
print("\nPreview:")
print(airdata_df.head())