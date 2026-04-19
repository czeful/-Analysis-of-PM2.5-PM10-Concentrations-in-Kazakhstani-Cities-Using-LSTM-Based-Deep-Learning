from pathlib import Path
import pandas as pd
import xarray as xr

ERA5_DIR = Path(r"C:\air_quality_project\processed\era5_unpacked")

def load_instant(fp):
    city = fp.name.split("_era5_")[0]
    ds = xr.open_dataset(fp)
    df = ds[["t2m"]].to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "datetime"})
    df["city"] = city
    return df[["datetime", "city"]]

def load_accum(fp):
    city = fp.name.split("_era5_")[0]
    ds = xr.open_dataset(fp)
    df = ds[["ssrd"]].to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "datetime"})
    df["city"] = city
    return df[["datetime", "city"]]

def load_t850(fp):
    city = fp.name.split("_era5_")[0]
    ds = xr.open_dataset(fp)
    df = ds[["t"]].to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "datetime"})
    df["city"] = city
    return df[["datetime", "city"]]

instant_files = sorted(ERA5_DIR.glob("*_surface_*_instant.nc"))
accum_files = sorted(ERA5_DIR.glob("*_surface_*_accum.nc"))
t850_files = sorted(ERA5_DIR.glob("*_t850_*.nc"))

print("instant files:", len(instant_files))
print("accum files:  ", len(accum_files))
print("t850 files:   ", len(t850_files))

instant_df = pd.concat([load_instant(fp) for fp in instant_files], ignore_index=True)
accum_df = pd.concat([load_accum(fp) for fp in accum_files], ignore_index=True)
t850_df = pd.concat([load_t850(fp) for fp in t850_files], ignore_index=True)

for name, df in [("instant", instant_df), ("accum", accum_df), ("t850", t850_df)]:
    print(f"\n--- {name.upper()} ---")
    print("rows by city:")
    print(df["city"].value_counts())

    dup = df.duplicated(subset=["city", "datetime"]).sum()
    print("duplicate city+datetime rows:", dup)

    print("duplicates by city:")
    dup_by_city = df.groupby("city").apply(
        lambda x: x.duplicated(subset=["city", "datetime"]).sum()
    )
    print(dup_by_city)

    print("unique datetimes by city:")
    uniq = df.groupby("city")["datetime"].nunique()
    print(uniq)