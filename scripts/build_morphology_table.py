from pathlib import Path
import pandas as pd

MORPH_DIR = Path(r"C:\air_quality_project\raw\morphology")
OUT_FILE = Path(r"C:\air_quality_project\processed\morphology_combined.csv")

def clean_city(df, col="city"):
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col].str.lower() != "city"].copy()
    return df

# 1. morphology_features.csv
features_fp = MORPH_DIR / "morphology_features.csv"
features = pd.read_csv(features_fp, header=None)
features = features.iloc[:, :4].copy()
features.columns = ["city", "tri", "ndvi", "rnd"]
features = clean_city(features)

# 2. tci_values.csv
tci_fp = MORPH_DIR / "tci_values.csv"
tci = pd.read_csv(tci_fp, header=None)
tci = tci.iloc[:, :2].copy()
tci.columns = ["city", "tci"]
tci = clean_city(tci)

# 3. d_ind_values.csv
dind_fp = MORPH_DIR / "d_ind_values.csv"
dind = pd.read_csv(dind_fp, header=None)
dind = dind.iloc[:, :2].copy()
dind.columns = ["city", "d_ind"]
dind = clean_city(dind)

# convert numeric columns
for col in ["tri", "ndvi", "rnd"]:
    features[col] = pd.to_numeric(features[col], errors="coerce")

tci["tci"] = pd.to_numeric(tci["tci"], errors="coerce")
dind["d_ind"] = pd.to_numeric(dind["d_ind"], errors="coerce")

# merge
morph_df = features.merge(tci, on="city", how="outer", validate="one_to_one")
morph_df = morph_df.merge(dind, on="city", how="outer", validate="one_to_one")

morph_df = morph_df.sort_values("city").reset_index(drop=True)

morph_df.to_csv(OUT_FILE, index=False)

print("Done")
print(f"Saved to: {OUT_FILE}")
print("Shape:", morph_df.shape)
print("\nColumns:")
print(morph_df.columns.tolist())
print("\nMissing values:")
print(morph_df.isna().sum())
print("\nPreview:")
print(morph_df)