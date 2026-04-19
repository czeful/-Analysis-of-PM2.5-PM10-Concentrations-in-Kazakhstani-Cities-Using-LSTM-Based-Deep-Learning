import xarray as xr

files = [
    r"C:\air_quality_project\processed\era5_unpacked\Karaganda_era5_surface_2020_01_instant.nc",
    r"C:\air_quality_project\processed\era5_unpacked\Karaganda_era5_surface_2020_01_accum.nc",
    r"C:\air_quality_project\processed\era5_unpacked\Karaganda_era5_t850_2020_01.nc",
]

for fp in files:
    print("\nFILE:", fp)
    ds = xr.open_dataset(fp)
    print(ds)
    print("dims:", ds.dims)
    if "latitude" in ds.coords:
        print("latitude values:", ds["latitude"].values)
    if "longitude" in ds.coords:
        print("longitude values:", ds["longitude"].values)