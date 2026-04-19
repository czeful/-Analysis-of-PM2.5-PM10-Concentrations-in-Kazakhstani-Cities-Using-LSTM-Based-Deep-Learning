import pandas as pd

file_path = r"C:\air_quality_project\processed\era5_combined.csv"
df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nCities:")
print(df["city"].value_counts())

print("\nDate range:")
print(df["datetime"].min(), "->", df["datetime"].max())