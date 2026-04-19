from pathlib import Path
import json
import pandas as pd

root = Path(r"C:\air_quality_project\processed\models_by_city")
rows = []

for city_dir in root.iterdir():
    if not city_dir.is_dir():
        continue

    metrics_file = city_dir / "lstm_transformer_metrics.json"
    if not metrics_file.exists():
        continue

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    city = data["city"]

    for split in ["train_metrics", "valid_metrics", "test_metrics"]:
        for pollutant in ["pm25", "pm10"]:
            rows.append({
                "city": city,
                "split": split.replace("_metrics", "").upper(),
                "pollutant": pollutant.upper(),
                "RMSE": data[split][pollutant]["RMSE"],
                "MAE": data[split][pollutant]["MAE"],
                "R2": data[split][pollutant]["R2"],
            })

df = pd.DataFrame(rows)
out_file = root / "all_city_metrics_summary.csv"
df.to_csv(out_file, index=False)

print("Saved:", out_file)
print(df)