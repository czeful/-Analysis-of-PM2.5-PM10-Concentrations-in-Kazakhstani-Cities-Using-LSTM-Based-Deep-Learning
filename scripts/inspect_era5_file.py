from pathlib import Path
import zipfile

file_path = Path(r"C:\air_quality_project\raw\era5\Almaty_era5_surface_2020_01.nc")

print("Exists:", file_path.exists())
print("Size:", file_path.stat().st_size, "bytes")

with open(file_path, "rb") as f:
    header = f.read(16)

print("First 16 bytes:", header)
print("Hex:", header.hex())

print("Is ZIP:", zipfile.is_zipfile(file_path))