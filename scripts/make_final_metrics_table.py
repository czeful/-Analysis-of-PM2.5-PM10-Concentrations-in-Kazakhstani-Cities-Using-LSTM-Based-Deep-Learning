from pathlib import Path
import json
import pandas as pd

BASE = Path(r"C:\air_quality_project\processed")

rows = []

# PM2.5 models
pm25_root = BASE / "models_pm25_by_city"
for city_dir in pm25_root.iterdir():
    if not city_dir.is_dir():
        continue

    fp = city_dir / "pm25_transformer_metrics.json"
    if not fp.exists():
        continue

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "city": data["city"],
        "target": "PM2.5",
        "split": "TRAIN",
        "RMSE": data["train_metrics"]["RMSE"],
        "MAE": data["train_metrics"]["MAE"],
        "R2": data["train_metrics"]["R2"],
    })
    rows.append({
        "city": data["city"],
        "target": "PM2.5",
        "split": "VALID",
        "RMSE": data["valid_metrics"]["RMSE"],
        "MAE": data["valid_metrics"]["MAE"],
        "R2": data["valid_metrics"]["R2"],
    })
    rows.append({
        "city": data["city"],
        "target": "PM2.5",
        "split": "TEST",
        "RMSE": data["test_metrics"]["RMSE"],
        "MAE": data["test_metrics"]["MAE"],
        "R2": data["test_metrics"]["R2"],
    })

# PM10 models
pm10_root = BASE / "models_pm10_by_city"
for city_dir in pm10_root.iterdir():
    if not city_dir.is_dir():
        continue

    fp = city_dir / "pm10_transformer_metrics.json"
    if not fp.exists():
        continue

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "city": data["city"],
        "target": "PM10",
        "split": "TRAIN",
        "RMSE": data["train_metrics"]["RMSE"],
        "MAE": data["train_metrics"]["MAE"],
        "R2": data["train_metrics"]["R2"],
    })
    rows.append({
        "city": data["city"],
        "target": "PM10",
        "split": "VALID",
        "RMSE": data["valid_metrics"]["RMSE"],
        "MAE": data["valid_metrics"]["MAE"],
        "R2": data["valid_metrics"]["R2"],
    })
    rows.append({
        "city": data["city"],
        "target": "PM10",
        "split": "TEST",
        "RMSE": data["test_metrics"]["RMSE"],
        "MAE": data["test_metrics"]["MAE"],
        "R2": data["test_metrics"]["R2"],
    })

df = pd.DataFrame(rows)
df = df.sort_values(["split", "city", "target"]).reset_index(drop=True)

out_dir = BASE / "final_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "final_metrics_table.csv"
df.to_csv(out_file, index=False)

print("Saved:", out_file)
print(df)