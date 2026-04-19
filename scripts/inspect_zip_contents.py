from pathlib import Path
import zipfile

file_path = Path(r"C:\air_quality_project\raw\era5\Almaty_era5_surface_2020_01.nc")

with zipfile.ZipFile(file_path, "r") as z:
    print("Archive name:", file_path.name)
    print("Files inside:")
    for name in z.namelist():
        print(" -", name)