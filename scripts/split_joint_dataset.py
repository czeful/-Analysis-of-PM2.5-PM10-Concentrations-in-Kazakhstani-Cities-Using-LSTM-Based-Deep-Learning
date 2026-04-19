from pathlib import Path
import pandas as pd

IN_FILE = Path(r"C:\air_quality_project\processed\training_dataset_joint_pm25_pm10.csv")

TRAIN_OUT = Path(r"C:\air_quality_project\processed\train_joint.csv")
VALID_OUT = Path(r"C:\air_quality_project\processed\valid_joint.csv")
TEST_OUT = Path(r"C:\air_quality_project\processed\test_joint.csv")

df = pd.read_csv(IN_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)
print("Full date range:", df["datetime"].min(), "->", df["datetime"].max())

# новый split
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

train = train.sort_values(["city", "datetime"]).reset_index(drop=True)
valid = valid.sort_values(["city", "datetime"]).reset_index(drop=True)
test = test.sort_values(["city", "datetime"]).reset_index(drop=True)

train.to_csv(TRAIN_OUT, index=False)
valid.to_csv(VALID_OUT, index=False)
test.to_csv(TEST_OUT, index=False)

print("\nDONE")
print(f"Train saved to: {TRAIN_OUT}")
print(f"Valid saved to: {VALID_OUT}")
print(f"Test saved to:  {TEST_OUT}")

print("\nShapes:")
print("Train:", train.shape)
print("Valid:", valid.shape)
print("Test:", test.shape)

print("\nRows by city in TRAIN:")
print(train["city"].value_counts())

print("\nRows by city in VALID:")
print(valid["city"].value_counts())

print("\nRows by city in TEST:")
print(test["city"].value_counts())

print("\nDate ranges:")
if len(train) > 0:
    print("Train:", train["datetime"].min(), "->", train["datetime"].max())
if len(valid) > 0:
    print("Valid:", valid["datetime"].min(), "->", valid["datetime"].max())
if len(test) > 0:
    print("Test:", test["datetime"].min(), "->", test["datetime"].max())