from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\training_pm10_transformer.csv")

TRAIN_OUT = Path(r"C:\air_quality_project\processed\train_pm10.csv")
VALID_OUT = Path(r"C:\air_quality_project\processed\valid_pm10.csv")
TEST_OUT = Path(r"C:\air_quality_project\processed\test_pm10.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)
print("Full date range:", df["datetime"].min(), "->", df["datetime"].max())

train = df[
    (df["datetime"] >= df["datetime"].min()) &
    (df["datetime"] < "2023-07-01")
].copy()

valid = df[
    (df["datetime"] >= "2023-07-01") &
    (df["datetime"] < "2024-01-01")
].copy()

test = df[
    (df["datetime"] >= "2024-01-01") &
    (df["datetime"] < "2025-01-01")
].copy()

for name, part in [("train", train), ("valid", valid), ("test", test)]:
    part.sort_values(["city", "datetime"], inplace=True)
    part.reset_index(drop=True, inplace=True)

train.to_csv(TRAIN_OUT, index=False)
valid.to_csv(VALID_OUT, index=False)
test.to_csv(TEST_OUT, index=False)

print("\nDONE")
print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test :", test.shape)

print("\nRows by city in TRAIN:")
print(train["city"].value_counts())

print("\nRows by city in VALID:")
print(valid["city"].value_counts())

print("\nRows by city in TEST:")
print(test["city"].value_counts())