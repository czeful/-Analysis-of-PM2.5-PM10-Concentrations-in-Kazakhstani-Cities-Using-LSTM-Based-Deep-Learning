from pathlib import Path
import re
import pandas as pd
import xarray as xr

ERA5_DIR = Path(r"C:\air_quality_project\processed\era5_unpacked")
OUT_FILE = Path(r"C:\air_quality_project\processed\era5_combined.csv")

surface_instant_files = sorted(ERA5_DIR.glob("*_surface_*_instant.nc"))
surface_accum_files = sorted(ERA5_DIR.glob("*_surface_*_accum.nc"))
t850_files = sorted(ERA5_DIR.glob("*_t850_*.nc"))


def parse_file_info(file_path: Path):
    """
    Example names:
    Almaty_era5_surface_2020_01_instant.nc
    Almaty_era5_surface_2020_01_accum.nc
    Almaty_era5_t850_2020_01.nc
    """
    name = file_path.stem

    m_surface = re.match(r"^(?P<city>.+?)_era5_surface_(?P<year>\d{4})_(?P<month>\d{2})_(?P<kind>instant|accum)$", name)
    if m_surface:
        return {
            "city": m_surface.group("city"),
            "year": int(m_surface.group("year")),
            "month": int(m_surface.group("month")),
            "dataset": "surface",
            "kind": m_surface.group("kind"),
        }

    m_t850 = re.match(r"^(?P<city>.+?)_era5_t850_(?P<year>\d{4})_(?P<month>\d{2})$", name)
    if m_t850:
        return {
            "city": m_t850.group("city"),
            "year": int(m_t850.group("year")),
            "month": int(m_t850.group("month")),
            "dataset": "t850",
            "kind": None,
        }

    raise ValueError(f"Unexpected filename format: {file_path.name}")
def collapse_spatial_dims(ds: xr.Dataset) -> xr.Dataset:
    dims_to_mean = [d for d in ["latitude", "longitude"] if d in ds.dims]
    if dims_to_mean:
        ds = ds.mean(dim=dims_to_mean, keep_attrs=True)

    if "pressure_level" in ds.dims and ds.sizes["pressure_level"] == 1:
        ds = ds.isel(pressure_level=0, drop=True)

    return ds

def ds_to_df_surface_instant(file_path: Path) -> pd.DataFrame:
    meta = parse_file_info(file_path)
    ds = xr.open_dataset(file_path)
    ds = collapse_spatial_dims(ds)

    df = ds[["t2m", "u10", "v10", "blh", "tcwv", "msl"]].to_dataframe().reset_index()

    df = df.rename(columns={"valid_time": "datetime"})
    df["city"] = meta["city"]

    return df[["datetime", "city", "t2m", "u10", "v10", "blh", "tcwv", "msl"]]


def ds_to_df_surface_accum(file_path: Path) -> pd.DataFrame:
    meta = parse_file_info(file_path)
    ds = xr.open_dataset(file_path)
    ds = collapse_spatial_dims(ds)

    df = ds[["ssrd"]].to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "datetime"})
    df["city"] = meta["city"]

    return df[["datetime", "city", "ssrd"]]

def ds_to_df_t850(file_path: Path) -> pd.DataFrame:
    meta = parse_file_info(file_path)
    ds = xr.open_dataset(file_path)
    ds = collapse_spatial_dims(ds)

    df = ds[["t"]].to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "datetime", "t": "t850"})
    df["city"] = meta["city"]

    return df[["datetime", "city", "t850"]]


print(f"Found surface instant files: {len(surface_instant_files)}")
print(f"Found surface accum files:   {len(surface_accum_files)}")
print(f"Found t850 files:            {len(t850_files)}")

# 1. Read all files
instant_parts = []
for fp in surface_instant_files:
    print(f"Reading instant: {fp.name}")
    instant_parts.append(ds_to_df_surface_instant(fp))

accum_parts = []
for fp in surface_accum_files:
    print(f"Reading accum:   {fp.name}")
    accum_parts.append(ds_to_df_surface_accum(fp))

t850_parts = []
for fp in t850_files:
    print(f"Reading t850:    {fp.name}")
    t850_parts.append(ds_to_df_t850(fp))

instant_df = pd.concat(instant_parts, ignore_index=True)
accum_df = pd.concat(accum_parts, ignore_index=True)
t850_df = pd.concat(t850_parts, ignore_index=True)

instant_df = instant_df.drop_duplicates(subset=["datetime", "city"])
accum_df = accum_df.drop_duplicates(subset=["datetime", "city"])
t850_df = t850_df.drop_duplicates(subset=["datetime", "city"])

# 2. Drop helper columns that are constant and not needed
# expver and number are already not included because we selected only needed variables

# 3. Merge instant + accum
era5_df = instant_df.merge(
    accum_df[["datetime", "city", "ssrd"]],
    on=["datetime", "city"],
    how="left",
    validate="one_to_one"
)

# 4. Merge + t850
era5_df = era5_df.merge(
    t850_df[["datetime", "city", "t850"]],
    on=["datetime", "city"],
    how="left",
    validate="one_to_one"
)
# 5. Sort
era5_df = era5_df.sort_values(["city", "datetime"]).reset_index(drop=True)

# 6. Save
era5_df.to_csv(OUT_FILE, index=False)

print("\nDone")
print(f"Saved to: {OUT_FILE}")
print(f"Shape: {era5_df.shape}")
print("\nColumns:")
print(list(era5_df.columns))
print("\nRows by city:")
print(era5_df["city"].value_counts())
print("\nPreview:")
print(era5_df.head())