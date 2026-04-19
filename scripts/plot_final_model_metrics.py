from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

IN_FILE = Path(r"C:\air_quality_project\processed\final_outputs\final_metrics_table.csv")
OUT_DIR = Path(r"C:\air_quality_project\processed\final_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_FILE)

# only TEST results
df = df[df["split"] == "TEST"].copy()

cities = ["Almaty", "Astana", "Karaganda"]

pm25 = df[df["target"] == "PM2.5"].set_index("city").reindex(cities)
pm10 = df[df["target"] == "PM10"].set_index("city").reindex(cities)

x = np.arange(len(cities))
width = 0.35

# R2
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, pm25["R2"], width, label="PM2.5")
plt.bar(x + width/2, pm10["R2"], width, label="PM10")
plt.xticks(x, cities)
plt.ylabel("R²")
plt.title("Test R² by City and Pollutant")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "final_test_r2.png", dpi=250, bbox_inches="tight")
plt.close()

# RMSE
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, pm25["RMSE"], width, label="PM2.5")
plt.bar(x + width/2, pm10["RMSE"], width, label="PM10")
plt.xticks(x, cities)
plt.ylabel("RMSE")
plt.title("Test RMSE by City and Pollutant")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "final_test_rmse.png", dpi=250, bbox_inches="tight")
plt.close()

# MAE
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, pm25["MAE"], width, label="PM2.5")
plt.bar(x + width/2, pm10["MAE"], width, label="PM10")
plt.xticks(x, cities)
plt.ylabel("MAE")
plt.title("Test MAE by City and Pollutant")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "final_test_mae.png", dpi=250, bbox_inches="tight")
plt.close()

print("Saved:")
print(OUT_DIR / "final_test_r2.png")
print(OUT_DIR / "final_test_rmse.png")
print(OUT_DIR / "final_test_mae.png")