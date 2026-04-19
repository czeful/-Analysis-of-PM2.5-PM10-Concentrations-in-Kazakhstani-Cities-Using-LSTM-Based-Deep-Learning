from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ======================
# FILES
# ======================
TRAIN_FILE = Path(r"C:\air_quality_project\processed\train_joint.csv")
VALID_FILE = Path(r"C:\air_quality_project\processed\valid_joint.csv")
TEST_FILE  = Path(r"C:\air_quality_project\processed\test_joint.csv")

OUT_ROOT = Path(r"C:\air_quality_project\processed\sequence_data_by_city")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LOOKBACK = 72

# ======================
# LOAD
# ======================
train = pd.read_csv(TRAIN_FILE)
valid = pd.read_csv(VALID_FILE)
test = pd.read_csv(TEST_FILE)

# ======================
# FEATURES / TARGETS
# ======================
feature_cols = [
    "co", "no2", "so2",
    "t2m", "u10", "v10", "blh", "tcwv", "msl", "ssrd", "t850",
    "aod_354", "co_col", "no2_trop", "so2_col",
    "tri", "ndvi", "rnd", "tci", "d_ind",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_24",
    "pm10_lag_1", "pm10_lag_3", "pm10_lag_24",
    "wind_speed", "wind_dir",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos"
]

target_cols = ["pm25", "pm10"]
cities = ["Almaty", "Astana", "Karaganda"]

def create_sequences(X, y, lookback):
    X_seq = []
    y_seq = []

    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

for city in cities:
    print(f"\nProcessing city: {city}")

    city_out = OUT_ROOT / city
    city_out.mkdir(parents=True, exist_ok=True)

    train_city = train[train["city"] == city].copy().sort_values("datetime").reset_index(drop=True)
    valid_city = valid[valid["city"] == city].copy().sort_values("datetime").reset_index(drop=True)
    test_city  = test[test["city"] == city].copy().sort_values("datetime").reset_index(drop=True)

    if len(train_city) <= LOOKBACK or len(valid_city) <= LOOKBACK or len(test_city) <= LOOKBACK:
        raise ValueError(f"Not enough rows for {city} after split")

    # fill remaining feature NaN using TRAIN medians of this city only
    train_medians = train_city[feature_cols].median(numeric_only=True)

    train_city[feature_cols] = train_city[feature_cols].fillna(train_medians)
    valid_city[feature_cols] = valid_city[feature_cols].fillna(train_medians)
    test_city[feature_cols]  = test_city[feature_cols].fillna(train_medians)

    # safety checks
    if train_city[feature_cols].isna().sum().sum() > 0:
        raise ValueError(f"NaN still present in train features for {city}")
    if valid_city[feature_cols].isna().sum().sum() > 0:
        raise ValueError(f"NaN still present in valid features for {city}")
    if test_city[feature_cols].isna().sum().sum() > 0:
        raise ValueError(f"NaN still present in test features for {city}")

    scaler = StandardScaler()

    X_train_raw = scaler.fit_transform(train_city[feature_cols]).astype(np.float32)
    X_valid_raw = scaler.transform(valid_city[feature_cols]).astype(np.float32)
    X_test_raw  = scaler.transform(test_city[feature_cols]).astype(np.float32)

    y_train_raw = train_city[target_cols].values.astype(np.float32)
    y_valid_raw = valid_city[target_cols].values.astype(np.float32)
    y_test_raw  = test_city[target_cols].values.astype(np.float32)

    X_train, y_train = create_sequences(X_train_raw, y_train_raw, LOOKBACK)
    X_valid, y_valid = create_sequences(X_valid_raw, y_valid_raw, LOOKBACK)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, LOOKBACK)

    np.save(city_out / "X_train.npy", X_train)
    np.save(city_out / "y_train.npy", y_train)
    np.save(city_out / "X_valid.npy", X_valid)
    np.save(city_out / "y_valid.npy", y_valid)
    np.save(city_out / "X_test.npy", X_test)
    np.save(city_out / "y_test.npy", y_test)

    joblib.dump(scaler, city_out / "feature_scaler.pkl")
    joblib.dump(train_medians, city_out / "feature_fill_medians.pkl")
    joblib.dump(feature_cols, city_out / "feature_cols.pkl")
    joblib.dump(target_cols, city_out / "target_cols.pkl")

    print(f"{city} shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

print("\nDONE")
print(f"Saved to: {OUT_ROOT}")