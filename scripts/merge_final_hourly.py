from pathlib import Path
import pandas as pd

AIR_FILE = Path(r"C:\air_quality_project\processed\airdata_article_2020_2024.csv")
ERA5_FILE = Path(r"C:\air_quality_project\processed\era5_combined.csv")
SENT_FILE = Path(r"C:\air_quality_project\processed\sentinel_hourly.csv")
MORPH_FILE = Path(r"C:\air_quality_project\processed\morphology_clean.csv")

OUT_FILE = Path(r"C:\air_quality_project\processed\final_hourly_dataset.csv")

# ======================
# 1. READ FILES
# ======================
air = pd.read_csv(AIR_FILE)
era5 = pd.read_csv(ERA5_FILE)
sent = pd.read_csv(SENT_FILE)
morph = pd.read_csv(MORPH_FILE)

air["datetime"] = pd.to_datetime(air["datetime"])
era5["datetime"] = pd.to_datetime(era5["datetime"])
sent["datetime"] = pd.to_datetime(sent["datetime"])

# ======================
# 2. BASIC CHECKS
# ======================
print("AIR shape:", air.shape)
print("ERA5 shape:", era5.shape)
print("SENT shape:", sent.shape)
print("MORPH shape:", morph.shape)

print("\nAIR duplicate keys:", air.duplicated(subset=["city", "datetime"]).sum())
print("ERA5 duplicate keys:", era5.duplicated(subset=["city", "datetime"]).sum())
print("SENT duplicate keys:", sent.duplicated(subset=["city", "datetime"]).sum())
print("MORPH duplicate cities:", morph.duplicated(subset=["city"]).sum())

# ======================
# 3. MERGE AIR + ERA5
# keep only rows where target observations exist
# ======================
df = air.merge(
    era5,
    on=["city", "datetime"],
    how="left",
    validate="many_to_one"
)

print("\nAfter AIR + ERA5:", df.shape)

# ======================
# 4. + SENTINEL
# keep all rows from current hourly target table
# ======================
df = df.merge(
    sent,
    on=["city", "datetime"],
    how="left",
    validate="many_to_one"
)

print("After + SENTINEL:", df.shape)

# ======================
# 5. + MORPHOLOGY
# static by city
# ======================
df = df.merge(
    morph,
    on="city",
    how="left",
    validate="many_to_one"
)

print("After + MORPH:", df.shape)

# ======================
# 6. FINAL SORT
# ======================
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

# ======================
# 7. SAVE
# ======================
df.to_csv(OUT_FILE, index=False)

print("\nDONE")
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