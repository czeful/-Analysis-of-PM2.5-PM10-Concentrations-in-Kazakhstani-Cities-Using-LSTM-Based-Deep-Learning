from pathlib import Path
import pandas as pd

root = Path(r"C:\air_quality_project\processed\shap_by_city")
rows = []

for city_dir in root.iterdir():
    if not city_dir.is_dir():
        continue

    city = city_dir.name

    for target in ["pm25", "pm10"]:
        fp = city_dir / f"{target}_shap_importance.csv"
        if not fp.exists():
            continue

        df = pd.read_csv(fp).head(10).copy()
        df["city"] = city
        df["target"] = target
        rows.append(df)

if not rows:
    raise ValueError("No SHAP files found")

out = pd.concat(rows, ignore_index=True)
out_file = root / "shap_top10_summary.csv"
out.to_csv(out_file, index=False)

print("Saved:", out_file)
print(out.head(30))