from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(r"C:\air_quality_project\processed")
OUT_DIR = BASE / "final_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cities = ["Almaty", "Astana", "Karaganda"]

def load_shap(city, target):
    fp = BASE / "shap_final" / city / f"{target}_shap.csv"
    df = pd.read_csv(fp)
    df["city"] = city
    df["target"] = target
    return df

for target in ["pm25", "pm10"]:
    frames = []
    for city in cities:
        frames.append(load_shap(city, target))

    full = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, city in zip(axes, cities):
        city_df = (
            full[full["city"] == city]
            .sort_values("normalized_importance", ascending=False)
            .head(5)
            .iloc[::-1]
        )

        ax.barh(city_df["feature"], city_df["normalized_importance"])
        ax.set_title(city)
        ax.set_xlabel("Normalized importance")

    fig.suptitle(f"Top SHAP Features for {target.upper()}", fontsize=14)
    plt.tight_layout()
    out_path = OUT_DIR / f"final_shap_{target}.png"
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)