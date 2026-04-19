from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_FILE = Path(r"C:\air_quality_project\processed\models_by_city\all_city_metrics_summary.csv")
OUT_DIR = Path(r"C:\air_quality_project\processed\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_FILE)

# оставим только TEST и R2
df_test = df[df["split"] == "TEST"].copy()

cities = ["Almaty", "Astana", "Karaganda"]
pollutants = ["PM25", "PM10"]

# нормализуем названия pollutant
df_test["pollutant"] = df_test["pollutant"].str.upper().str.replace(".", "", regex=False)

for metric in ["R2", "RMSE", "MAE"]:
    pivot = df_test.pivot(index="city", columns="pollutant", values=metric).reindex(cities)

    plt.figure(figsize=(8, 5))
    x = range(len(pivot.index))
    width = 0.35

    plt.bar([i - width/2 for i in x], pivot["PM25"], width=width, label="PM2.5")
    plt.bar([i + width/2 for i in x], pivot["PM10"], width=width, label="PM10")

    plt.xticks(list(x), pivot.index)
    plt.ylabel(metric)
    plt.title(f"Test {metric} by City")
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"model_test_{metric.lower()}_by_city.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)

print("Done")