from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\training_pm25_transformer.csv")
OUT_FILE = Path(r"C:\air_quality_project\processed\training_pm25_transformer_clean_astana.csv")

df = pd.read_csv(IN_FILE)

# split cities
astana = df[df["city"] == "Astana"].copy()
others = df[df["city"] != "Astana"].copy()

print("Before:", len(astana))

# фильтр
astana = astana[astana["pm25"] < 200]

print("After:", len(astana))

df_clean = pd.concat([others, astana]).sort_values(["city", "datetime"])

df_clean.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)