from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\morphology_combined.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\morphology_clean.csv")

df = pd.read_csv(IN_FILE)

# clean city names
df["city"] = df["city"].astype(str).str.strip()

# remove fake header row if present
df = df[df["city"].str.lower() != "city"].copy()

# keep only real cities
valid_cities = ["Almaty", "Astana", "Karaganda"]
df = df[df["city"].isin(valid_cities)].copy()

# convert numeric columns
for col in ["tri", "ndvi", "rnd", "tci", "d_ind"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("city").reset_index(drop=True)

df.to_csv(OUT_FILE, index=False)

print("Done")
print(f"Saved to: {OUT_FILE}")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())
print("\nPreview:")
print(df)