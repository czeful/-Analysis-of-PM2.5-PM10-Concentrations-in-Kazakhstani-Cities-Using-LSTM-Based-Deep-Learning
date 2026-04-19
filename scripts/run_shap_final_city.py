from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys

# ВАЖНО ДЛЯ SHAP + LSTM
torch.backends.cudnn.enabled = False

# ======================
# INPUT
# ======================
if len(sys.argv) < 3:
    raise ValueError("Usage: python run_shap_final_city.py <CITY> <TARGET>")

CITY = sys.argv[1]        # Almaty / Astana / Karaganda
TARGET = sys.argv[2]      # pm25 / pm10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PATHS
# ======================
BASE = Path(r"C:\air_quality_project\processed")

if TARGET == "pm25":
    DATA_DIR = BASE / "sequence_data_pm25_by_city" / CITY
    MODEL_DIR = BASE / "models_pm25_by_city" / CITY
    MODEL_FILE = MODEL_DIR / "pm25_transformer_best.pt"
else:
    DATA_DIR = BASE / "sequence_data_pm10_by_city" / CITY
    MODEL_DIR = BASE / "models_pm10_by_city" / CITY
    MODEL_FILE = MODEL_DIR / "pm10_transformer_best.pt"

OUT_DIR = BASE / "shap_final" / CITY
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# CONFIG
# ======================
BACKGROUND_SIZE = 50
TEST_SAMPLE_SIZE = 100
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================
# LOAD DATA
# ======================
X = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
feature_cols = joblib.load(DATA_DIR / "feature_cols.pkl")

# берём только часть
sample_size = min(TEST_SAMPLE_SIZE, len(X))
background_size = min(BACKGROUND_SIZE, len(X))

X_sample = X[:sample_size]
X_background = X[:background_size]

print("City:", CITY)
print("Target:", TARGET)
print("Data shape:", X_sample.shape)
print("Background shape:", X_background.shape)

# ======================
# MODEL
# ======================
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_proj = nn.Linear(config["input_size"], config["d_model"])

        self.lstm = nn.LSTM(
            input_size=config["d_model"],
            hidden_size=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            batch_first=True,
            dropout=config["dropout"] if config["lstm_layers"] > 1 else 0.0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["lstm_hidden"],
            nhead=config["n_heads"],
            dim_feedforward=config["ff_dim"],
            dropout=config["dropout"],
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["transformer_layers"]
        )

        self.head = nn.Sequential(
            nn.Linear(config["lstm_hidden"], 64),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(64, config["output_size"])
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.head(x)

# ======================
# LOAD MODEL
# ======================
checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
config = checkpoint["config"]

model = Model(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# ВАЖНО: train mode для backward через LSTM
model.train()

print("Model loaded from:", MODEL_FILE)

# ======================
# SHAP
# ======================
X_background_tensor = torch.from_numpy(X_background).to(device)
X_sample_tensor = torch.from_numpy(X_sample).to(device)

explainer = shap.GradientExplainer(model, X_background_tensor)
shap_values = explainer.shap_values(X_sample_tensor)

# ======================
# NORMALIZE SHAP OUTPUT SHAPE
# ======================
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.array(shap_values)

# possible shapes:
# (samples, seq, features, 1)
# (samples, seq, features)
if shap_values.ndim == 4:
    shap_values = shap_values[..., 0]

# aggregate over time
# -> shape becomes (samples, features)
shap_mean_over_time = np.mean(np.abs(shap_values), axis=1)

# aggregate over samples
# -> shape becomes (features,)
importance = shap_mean_over_time.mean(axis=0)

df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False).reset_index(drop=True)

# normalized importance
df["normalized_importance"] = df["importance"] / df["importance"].sum()

# ======================
# SAVE TABLE
# ======================
csv_path = OUT_DIR / f"{TARGET}_shap.csv"
df.to_csv(csv_path, index=False)

print("Saved:", csv_path)

# ======================
# PLOT
# ======================
top = df.head(15).iloc[::-1]

plt.figure(figsize=(9, 6))
plt.barh(top["feature"], top["normalized_importance"])
plt.xlabel("Normalized mean |SHAP|")
plt.ylabel("Feature")
plt.title(f"{CITY} - {TARGET.upper()} SHAP Importance")
plt.tight_layout()

png_path = OUT_DIR / f"{TARGET}_shap.png"
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved plot:", png_path)

print("\nTOP FEATURES:")
print(df.head(10))