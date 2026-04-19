from pathlib import Path
import zipfile
import shutil

src_dir = Path(r"C:\air_quality_project\raw\era5")
dst_dir = Path(r"C:\air_quality_project\processed\era5_unpacked")
dst_dir.mkdir(parents=True, exist_ok=True)

archive_files = sorted(src_dir.glob("*.nc"))
print(f"Found {len(archive_files)} archive-like files")

for archive_path in archive_files:
    try:
        if not zipfile.is_zipfile(archive_path):
            print(f"SKIP not zip: {archive_path.name}")
            continue

        base_name = archive_path.stem  # example: Almaty_era5_surface_2020_01

        with zipfile.ZipFile(archive_path, "r") as z:
            inner_files = z.namelist()
            print(f"\nProcessing: {archive_path.name}")
            print("Inside:", inner_files)

            for member in inner_files:
                member_name = Path(member).name.lower()

                if member_name.endswith(".nc"):
                    if "instant" in member_name:
                        new_name = f"{base_name}_instant.nc"
                    elif "accum" in member_name:
                        new_name = f"{base_name}_accum.nc"
                    else:
                        new_name = f"{base_name}.nc"

                    temp_extract_path = dst_dir / Path(member).name
                    final_path = dst_dir / new_name

                    with z.open(member) as source, open(final_path, "wb") as target:
                        shutil.copyfileobj(source, target)

                    print(f"Saved: {final_path.name}")
                else:
                    print(f"Skipped non-nc: {member}")

    except Exception as e:
        print(f"ERROR in {archive_path.name}: {e}")

print("\nDone")