import xarray as xr

file_path = r"C:\air_quality_project\raw\era5\Almaty_era5_surface_2020_01.nc"

ds = xr.open_dataset(file_path, engine="netcdf4")

print(ds)