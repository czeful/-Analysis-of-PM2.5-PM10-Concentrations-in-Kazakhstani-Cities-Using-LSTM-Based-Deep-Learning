from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\final_hourly_fullgrid.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\final_hourly_clean.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

# ======================
# 1. SORT
# ======================
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

# ======================
# 2. DEFINE GROUPS
# ======================

meteo_cols = ["t2m", "u10", "v10", "blh", "tcwv", "msl", "ssrd", "t850"]
sentinel_cols = ["aod_354", "co_col", "no2_trop", "so2_col"]
morph_cols = ["tri", "ndvi", "rnd", "tci", "d_ind"]

# ======================
# 3. APPLY PER CITY
# ======================

result_parts = []

for city in df["city"].unique():
    print(f"Processing {city}")

    city_df = df[df["city"] == city].copy()

    # --- METEO: interpolation ---
    city_df[meteo_cols] = city_df[meteo_cols].interpolate(method="linear")

    # --- SENTINEL: forward fill ---
    city_df[sentinel_cols] = city_df[sentinel_cols].ffill()

    # --- MORPHOLOGY: forward fill ---
    city_df[morph_cols] = city_df[morph_cols].ffill()

    result_parts.append(city_df)

df_clean = pd.concat(result_parts, ignore_index=True)
df_clean = df_clean.sort_values(["city", "datetime"]).reset_index(drop=True)

# ======================
# 4. SAVE
# ======================
df_clean.to_csv(OUT_FILE, index=False)

print("\nDONE")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean.isna().sum())

print("\nPreview:")
print(df_clean.head())