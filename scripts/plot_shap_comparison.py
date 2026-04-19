from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN_FILE = Path(r"C:\air_quality_project\processed\shap_by_city\shap_top10_summary.csv")
OUT_DIR = Path(r"C:\air_quality_project\processed\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_FILE)

cities = ["Almaty", "Astana", "Karaganda"]

for target in ["pm25", "pm10"]:
    target_df = df[df["target"] == target].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=False)

    for ax, city in zip(axes, cities):
        city_df = (
            target_df[target_df["city"] == city]
            .sort_values("normalized_importance", ascending=False)
            .head(5)
            .iloc[::-1]
        )

        ax.barh(city_df["feature"], city_df["normalized_importance"])
        ax.set_title(city)
        ax.set_xlabel("Normalized mean |SHAP|")

    fig.suptitle(f"Top SHAP Features for {target.upper()}", fontsize=14)
    plt.tight_layout()

    out_path = OUT_DIR / f"shap_top5_{target}_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)

print("Done")
