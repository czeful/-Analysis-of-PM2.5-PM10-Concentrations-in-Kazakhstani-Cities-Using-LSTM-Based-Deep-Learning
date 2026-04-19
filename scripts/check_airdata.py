import pandas as pd

files = [
    r"C:\air_quality_project\raw\air_data\al\pm25.csv",
    r"C:\air_quality_project\raw\air_data\as\pm25.csv",
    r"C:\air_quality_project\raw\air_data\krg\pm25.csv",
]

for fp in files:
    print("\nFILE:", fp)
    df = pd.read_csv(fp)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print("Shape:", df.shape)