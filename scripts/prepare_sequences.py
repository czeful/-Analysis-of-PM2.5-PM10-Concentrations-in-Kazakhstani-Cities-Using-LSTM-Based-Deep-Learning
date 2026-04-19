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

OUT_DIR = Path(r"C:\air_quality_project\processed\sequence_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 72

# ======================
# LOAD
# ======================
train = pd.read_csv(TRAIN_FILE)
valid = pd.read_csv(VALID_FILE)
test  = pd.read_csv(TEST_FILE)

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

# ======================
# SAFETY CHECKS
# ======================
for name, df in [("train", train), ("valid", valid), ("test", test)]:
    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_features or missing_targets:
        raise ValueError(
            f"{name} missing columns. Features: {missing_features}, Targets: {missing_targets}"
        )

# ======================
# IMPUTE FEATURE NaN USING TRAIN MEDIANS
# ======================
train_medians = train[feature_cols].median(numeric_only=True)

train[feature_cols] = train[feature_cols].fillna(train_medians)
valid[feature_cols] = valid[feature_cols].fillna(train_medians)
test[feature_cols]  = test[feature_cols].fillna(train_medians)

print("\nRemaining NaN after feature imputation:")
print("Train:", train[feature_cols].isna().sum().sum())
print("Valid:", valid[feature_cols].isna().sum().sum())
print("Test :", test[feature_cols].isna().sum().sum())

# ======================
# SCALING
# ======================
scaler = StandardScaler()

train_features = scaler.fit_transform(train[feature_cols])
valid_features = scaler.transform(valid[feature_cols])
test_features  = scaler.transform(test[feature_cols])

train_targets = train[target_cols].values.astype(np.float32)
valid_targets = valid[target_cols].values.astype(np.float32)
test_targets  = test[target_cols].values.astype(np.float32)

# extra safety
if np.isnan(train_features).any() or np.isnan(valid_features).any() or np.isnan(test_features).any():
    raise ValueError("NaN detected in scaled features")

if np.isnan(train_targets).any() or np.isnan(valid_targets).any() or np.isnan(test_targets).any():
    raise ValueError("NaN detected in targets")

# ======================
# SEQUENCE FUNCTION
# ======================
def create_sequences(X, y, lookback):
    X_seq = []
    y_seq = []

    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

# ======================
# APPLY PER CITY
# ======================
def build_dataset(df, X, y, lookback):
    X_all = []
    y_all = []

    for city in df["city"].unique():
        mask = df["city"] == city
        X_city = X[mask]
        y_city = y[mask]

        X_seq, y_seq = create_sequences(X_city, y_city, lookback)

        if len(X_seq) > 0:
            X_all.append(X_seq)
            y_all.append(y_seq)

    if not X_all:
        raise ValueError("No sequences were created")

    X_out = np.concatenate(X_all, axis=0)
    y_out = np.concatenate(y_all, axis=0)

    if np.isnan(X_out).any():
        raise ValueError("NaN detected in final X sequences")
    if np.isnan(y_out).any():
        raise ValueError("NaN detected in final y sequences")

    return X_out, y_out

# ======================
# BUILD
# ======================
X_train, y_train = build_dataset(train, train_features, train_targets, LOOKBACK)
X_valid, y_valid = build_dataset(valid, valid_features, valid_targets, LOOKBACK)
X_test, y_test = build_dataset(test, test_features, test_targets, LOOKBACK)

# ======================
# SAVE ARRAYS
# ======================
np.save(OUT_DIR / "X_train.npy", X_train)
np.save(OUT_DIR / "y_train.npy", y_train)

np.save(OUT_DIR / "X_valid.npy", X_valid)
np.save(OUT_DIR / "y_valid.npy", y_valid)

np.save(OUT_DIR / "X_test.npy", X_test)
np.save(OUT_DIR / "y_test.npy", y_test)

joblib.dump(scaler, OUT_DIR / "feature_scaler.pkl")
joblib.dump(train_medians, OUT_DIR / "feature_fill_medians.pkl")
joblib.dump(feature_cols, OUT_DIR / "feature_cols.pkl")
joblib.dump(target_cols, OUT_DIR / "target_cols.pkl")

# ======================
# OUTPUT
# ======================
print("\nFINAL DATA SHAPES:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_valid:", X_valid.shape)
print("y_valid:", y_valid.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nSaved files:")
print(OUT_DIR / "X_train.npy")
print(OUT_DIR / "y_train.npy")
print(OUT_DIR / "X_valid.npy")
print(OUT_DIR / "y_valid.npy")
print(OUT_DIR / "X_test.npy")
print(OUT_DIR / "y_test.npy")
print(OUT_DIR / "feature_scaler.pkl")
print(OUT_DIR / "feature_fill_medians.pkl")