from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import sys

torch.backends.cudnn.enabled = False
# ======================
# ARGUMENT
# ======================
if len(sys.argv) < 2:
    CITY = "Karaganda"
else:
    CITY = sys.argv[1]

# ======================
# PATHS
# ======================
SEQ_DIR = Path(r"C:\air_quality_project\processed\sequence_data_by_city") / CITY
MODEL_DIR = Path(r"C:\air_quality_project\processed\models_by_city") / CITY
OUT_DIR = Path(r"C:\air_quality_project\processed\shap_by_city") / CITY
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_FILE = SEQ_DIR / "X_train.npy"
X_TEST_FILE = SEQ_DIR / "X_test.npy"
FEATURE_COLS_FILE = SEQ_DIR / "feature_cols.pkl"

MODEL_FILE = MODEL_DIR / "lstm_transformer_best.pt"

# ======================
# CONFIG
# ======================
BACKGROUND_SIZE = 50
TEST_SAMPLE_SIZE = 100
BATCH_EXPLAIN_SIZE = 100
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("City:", CITY)

# ======================
# LOAD DATA
# ======================
X_train = np.load(X_TRAIN_FILE).astype(np.float32)
X_test = np.load(X_TEST_FILE).astype(np.float32)
feature_cols = joblib.load(FEATURE_COLS_FILE)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Number of features:", len(feature_cols))

# sample background and test subset
train_idx = np.random.choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False)
test_idx = np.random.choice(len(X_test), size=min(TEST_SAMPLE_SIZE, len(X_test)), replace=False)

X_background = X_train[train_idx]
X_explain = X_test[test_idx]

# ======================
# MODEL DEFINITION
# ======================
class LSTMTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=128,
        lstm_hidden=128,
        lstm_layers=2,
        n_heads=8,
        transformer_layers=2,
        ff_dim=256,
        dropout=0.2,
        output_size=2
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.head(x)
        return out

# ======================
# LOAD MODEL
# ======================
checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
config = checkpoint["config"]

model = LSTMTransformer(
    input_size=config["input_size"],
    d_model=config["d_model"],
    lstm_hidden=config["lstm_hidden"],
    lstm_layers=config["lstm_layers"],
    n_heads=config["n_heads"],
    transformer_layers=config["transformer_layers"],
    ff_dim=config["ff_dim"],
    dropout=config["dropout"],
    output_size=config["output_size"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded from:", MODEL_FILE)

# ======================
# WRAPPER FOR ONE TARGET
# ======================
class TargetWrapper(nn.Module):
    def __init__(self, base_model, target_idx):
        super().__init__()
        self.base_model = base_model
        self.target_idx = target_idx

    def forward(self, x):
        out = self.base_model(x)
        return out[:, self.target_idx:self.target_idx + 1]

# ======================
# SHAP FUNCTION
# ======================
def run_shap_for_target(target_idx, target_name):
    print(f"\nRunning SHAP for {target_name}...")

    wrapped_model = TargetWrapper(model, target_idx).to(device)
    wrapped_model.train()

    background_tensor = torch.tensor(X_background, dtype=torch.float32).to(device)
    explain_tensor = torch.tensor(X_explain, dtype=torch.float32).to(device)

    explainer = shap.GradientExplainer(wrapped_model, background_tensor)

    shap_values = explainer.shap_values(explain_tensor)

    # shap output may be list or array depending on version
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # expected shape: (samples, timesteps, features, 1) or (samples, timesteps, features)
    shap_values = np.array(shap_values)

    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    # aggregate absolute SHAP across time and samples
    mean_abs_shap_by_feature = np.mean(np.abs(shap_values), axis=(0, 1))

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap_by_feature
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["normalized_importance"] = (
        importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum()
    )

    csv_path = OUT_DIR / f"{target_name}_shap_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    print(f"Saved importance table: {csv_path}")
    print("\nTop 10 features:")
    print(importance_df.head(10))

    # bar plot top 15
    top_n = 15
    plot_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["normalized_importance"])
    plt.xlabel("Normalized mean |SHAP|")
    plt.ylabel("Feature")
    plt.title(f"{CITY} - {target_name} SHAP Feature Importance")
    plt.tight_layout()

    plot_path = OUT_DIR / f"{target_name}_shap_bar.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {plot_path}")

    return importance_df

# ======================
# RUN FOR BOTH TARGETS
# ======================
pm25_df = run_shap_for_target(0, "pm25")
pm10_df = run_shap_for_target(1, "pm10")

print("\nDONE")
print("Output folder:", OUT_DIR)