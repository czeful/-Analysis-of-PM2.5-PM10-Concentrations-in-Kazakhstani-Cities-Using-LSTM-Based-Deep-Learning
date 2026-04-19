import pandas as pd

files = [
    r"C:\air_quality_project\raw\sentinel\almaty_AOD_354.csv",
    r"C:\air_quality_project\raw\sentinel\almaty_CO_col.csv",
    r"C:\air_quality_project\raw\sentinel\almaty_NO2_trop.csv",
    r"C:\air_quality_project\raw\sentinel\almaty_SO2_col.csv",
]

for fp in files:
    print("\nFILE:", fp)
    df = pd.read_csv(fp)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print("Shape:", df.shape)