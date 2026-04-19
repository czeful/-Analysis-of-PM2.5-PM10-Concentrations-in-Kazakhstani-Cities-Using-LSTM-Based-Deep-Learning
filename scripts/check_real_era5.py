import xarray as xr

file_path = r"C:\air_quality_project\processed\era5_unpacked\Almaty_era5_surface_2020_01_instant.nc"
ds = xr.open_dataset(file_path)

print(ds)
print("\nVariables:")
print(list(ds.data_vars))