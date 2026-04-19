from pathlib import Path
import joblib

root = Path(r"C:\air_quality_project\processed\sequence_data_by_city")

for city in ["Almaty", "Astana", "Karaganda"]:
    fp = root / city / "feature_cols.pkl"
    cols = joblib.load(fp)
    print(f"\n{city}:")
    for c in cols:
        print(c)